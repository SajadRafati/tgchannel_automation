import argparse, hashlib, json, logging, os, re, sqlite3, sys, threading, time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from google import genai
from google.genai import types, errors as gerrors

ap = argparse.ArgumentParser()
ap.add_argument("--test", action="store_true", help="publish every few minutes")
args = ap.parse_args()

load_dotenv()
cfg = json.load(open("config.json", encoding="utf-8"))
TEST = args.test or cfg.get("test_mode", False)

# ---------- logging: console + file ----------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
root = logging.getLogger()
root.setLevel(logging.INFO)
fh = logging.FileHandler("bot.log", encoding="utf-8"); fh.setFormatter(fmt)
sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
root.addHandler(fh); root.addHandler(sh)
for n in ("httpx", "httpcore", "google_genai", "google_genai.models",
          "apscheduler", "urllib3"):
    logging.getLogger(n).setLevel(logging.WARNING)
log = logging.getLogger("bot")

if cfg.get("proxy"):
    os.environ["HTTP_PROXY"] = os.environ["HTTPS_PROXY"] = cfg["proxy"]

BOT_TOKEN = os.environ["BOT_TOKEN"]
ai = genai.Client(api_key=os.environ["GEMINI_KEY"])
MODELS = cfg.get("models") or [cfg["model"]]
lock = threading.RLock()
UA = {"User-Agent": "Mozilla/5.0"}

# ---------- database + memory ----------
db = sqlite3.connect("queue.db", check_same_thread=False)
db.execute("""CREATE TABLE IF NOT EXISTS q(
  id INTEGER PRIMARY KEY AUTOINCREMENT, src TEXT, msg_id INTEGER,
  text TEXT, hash TEXT UNIQUE, ts REAL,
  status TEXT DEFAULT 'new',
  UNIQUE(src, msg_id))""")
db.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
db.commit()

def meta_get(k, default=0):
    r = db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
    return r[0] if r else default

def meta_set(k, v):
    db.execute("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)", (k, str(v)))
    db.commit()

def status_line():
    rows = db.execute("SELECT status, COUNT(*) FROM q GROUP BY status").fetchall()
    return " ".join(f"{s}={n}" for s, n in rows) or "empty"

def hhmm(t):
    return datetime.fromtimestamp(t).strftime("%H:%M:%S")

# ---------- Gemini with fallback / backoff ----------
PROMPT = """You classify Persian Telegram posts. For each post decide keep=true ONLY if it is:
- an official news/announcement (circulars, price changes, rules, important events), or
- a registration/sign-up announcement (e.g. car sales registration, lottery, deadlines).
Ads, greetings, chit-chat, promos, polls and generic content => keep=false.
Return ONLY a JSON array with one entry per post: [{"id": <int>, "keep": true|false}]

Posts:
"""
blocked = {}          # model -> unix time until which it is skipped
ai_pause_until = 0.0
fail_streak = 0

def parse_retry(msg):
    m = re.search(r"retry in ([\d.]+)s", msg, re.I) or \
        re.search(r"retryDelay'?\"?: ?'?\"?(\d+)s", msg)
    return float(m.group(1)) if m else None

def call_ai(prompt):
    """Returns (text, model) or (None, None). Never raises."""
    global ai_pause_until, fail_streak
    now = time.time()
    if now < ai_pause_until:
        log.info("AI paused until %s", hhmm(ai_pause_until))
        return None, None
    available = [m for m in MODELS if blocked.get(m, 0) <= now]
    if not available:
        ai_pause_until = min(blocked.values())
        log.warning("All models limited/unavailable. Next try at %s", hhmm(ai_pause_until))
        return None, None
    for m in available:
        try:
            r = ai.models.generate_content(
                model=m, contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0))
            fail_streak = 0
            return r.text, m
        except gerrors.APIError as e:
            code, msg = getattr(e, "code", None), str(e)
            if code == 429:
                per_day = "PerDay" in msg or "per day" in msg.lower()
                wait = 3 * 3600 if per_day else (parse_retry(msg) or 60) + 5
                blocked[m] = now + wait
                log.warning("[%s] quota limit (429) -> skipping this model until %s",
                            m, hhmm(blocked[m]))
            elif code in (500, 502, 503, 504):
                wait = min(120 * (2 ** fail_streak), 1800)
                fail_streak += 1
                blocked[m] = now + wait
                log.warning("[%s] Google overloaded (%s) -> skipping until %s",
                            m, code, hhmm(blocked[m]))
            else:  # 400/401/403/404: bad model name or key
                blocked[m] = now + 3600
                log.error("[%s] API error %s: %s -> skipping for 1h (check model name/key)",
                          m, code, msg[:150])
        except Exception as e:  # network down etc.
            ai_pause_until = now + 120
            log.error("Cannot reach Gemini (VPN?): %s", str(e)[:150])
            return None, None
    return None, None

def classify_pending():
    """Classify one batch. Returns number of posts decided (0 = nothing/AI unavailable)."""
    db.execute("UPDATE q SET status='expired' WHERE status='new' AND ts<?",
               (time.time() - cfg["max_age_hours"] * 3600,))
    db.commit()
    rows = db.execute("SELECT id,text FROM q WHERE status='new' ORDER BY ts LIMIT ?",
                      (cfg["batch_size"],)).fetchall()
    if not rows:
        return 0
    payload = "\n".join(f"### ID {rid}\n{text[:700]}\n" for rid, text in rows)
    out, model = call_ai(PROMPT + payload)
    if out is None:
        return 0
    try:
        data = json.loads(re.sub(r"```(?:json)?", "", out).strip())
        if isinstance(data, dict):
            data = data.get("results") or data.get("posts") or []
        valid = {rid for rid, _ in rows}
        kept = rejected = 0
        for item in data:
            rid = int(item["id"])
            if rid not in valid:
                continue
            ok = bool(item.get("keep"))
            db.execute("UPDATE q SET status=? WHERE id=?",
                       ("queued" if ok else "rejected", rid))
            kept += ok
            rejected += (not ok)
        db.commit()
        log.info("AI [%s]: %d posts -> %d queued, %d rejected", model, kept + rejected,
                 kept, rejected)
        return kept + rejected
    except Exception as e:
        log.error("Bad AI response, will retry later: %s | %s", e, out[:150])
        return 0

# ---------- scraping with memory ----------
def norm_hash(text):
    return hashlib.md5(re.sub(r"\W+", "", text).encode()).hexdigest()

def parse_page(html):
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for w in soup.select("div.tgme_widget_message"):
        post = w.get("data-post")
        tm = w.select_one("a.tgme_widget_message_date time")
        if not (post and tm):
            continue
        try:
            ts = datetime.fromisoformat(tm["datetime"]).timestamp()
        except Exception:
            continue
        text = None
        box = w.select_one("div.tgme_widget_message_text")
        if box:
            for br in box.find_all("br"):
                br.replace_with("\n")
            for a in box.find_all("a"):
                label = a.get_text()
                a.replace_with(label if label.startswith("#")
                               else f"[{label}]({a.get('href', '')})")
            text = box.get_text()
        out.append({"id": int(post.split("/")[1]), "ts": ts, "text": text})
    return out

def fetch_page(ch, before=None):
    url = f"https://t.me/s/{ch}" + (f"?before={before}" if before else "")
    r = requests.get(url, timeout=20, headers=UA)
    r.raise_for_status()
    return parse_page(r.text)

def collect():
    since = time.time() - cfg["backfill_hours"] * 3600
    for ch in cfg["sources"]:
        try:
            last = int(meta_get(f"last:{ch}", 0))
            top, added, dups, before, reached = last, 0, 0, None, False
            for _ in range(cfg["max_pages"]):
                items = fetch_page(ch, before)
                if not items:
                    reached = True
                    break
                for it in items:
                    top = max(top, it["id"])
                    t = it["text"]
                    if it["id"] <= last or it["ts"] < since or not t or len(t) < 30:
                        continue
                    try:
                        db.execute("INSERT INTO q(src,msg_id,text,hash,ts) VALUES(?,?,?,?,?)",
                                   (ch, it["id"], t, norm_hash(t), it["ts"]))
                        added += 1
                    except sqlite3.IntegrityError:
                        dups += 1
                lo = min(items, key=lambda i: i["id"])
                if lo["id"] <= last + 1 or lo["ts"] < since:
                    reached = True
                    break
                before = lo["id"]
            db.commit()
            meta_set(f"last:{ch}", top)
            if not reached:
                log.warning("%s: more posts than max_pages covers; some may be missed", ch)
            log.info("crawl %-18s last_id=%s | new=%d dup=%d", ch, top, added, dups)
        except requests.RequestException as e:
            log.error("crawl %s failed (t.me unreachable? VPN/proxy): %s", ch, str(e)[:120])
        except Exception:
            log.exception("crawl %s crashed", ch)

# ---------- editing + publishing ----------
def clean(text):
    keep = []
    for line in text.splitlines():
        if re.search(r"https?://|t\.me/|\]\(|@\w+", line):
            continue
        keep.append(line.rstrip())
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(keep)).strip()
    return f"{body}\n\n{cfg['footer']}"[:4000]

def send(text):
    r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": cfg["target"], "text": text,
                            "disable_web_page_preview": True}, timeout=30)
    if not r.ok:
        raise RuntimeError(f"Telegram {r.status_code}: {r.text[:200]}")

def publish(slot=True):
    with lock:
        db.execute("UPDATE q SET status='expired' WHERE status='queued' AND ts<?",
                   (time.time() - cfg["max_age_hours"] * 3600,))
        row = db.execute("SELECT id,text FROM q WHERE status='queued' "
                         "ORDER BY ts LIMIT 1").fetchone()
        db.commit()
        if not row:
            pending = db.execute("SELECT COUNT(*) FROM q WHERE status='new'").fetchone()[0]
            if slot and pending:
                owed = min(int(meta_get("owed", 0)) + 1, cfg["max_owed"])
                meta_set("owed", owed)
                log.warning("Publish slot: nothing classified yet (%d unclassified). "
                            "Owed slots=%d, will post when AI is back.", pending, owed)
            elif slot:
                log.info("Publish slot: queue empty, nothing to post.")
            return False
        rid, text = row
        body = clean(text)
        if body.strip() == cfg["footer"]:
            db.execute("UPDATE q SET status='rejected' WHERE id=?", (rid,))
            db.commit()
            return False
        try:
            send(body)
        except Exception as e:
            log.error("Send failed (will retry next slot): %s", e)
            return False
        db.execute("UPDATE q SET status='posted' WHERE id=?", (rid,))
        db.commit()
        log.info("POSTED id=%s | %s", rid, body.replace("\n", " ")[:60])
        return True

def pay_owed():
    while int(meta_get("owed", 0)) > 0:
        if not publish(slot=False):
            break
        meta_set("owed", int(meta_get("owed", 0)) - 1)
        log.info("Paid one owed slot, remaining=%s", meta_get("owed", 0))
        time.sleep(8)

def poll():
    try:
        with lock:
            collect()
            for _ in range(10):
                if classify_pending() == 0:
                    break
                time.sleep(cfg.get("ai_gap_seconds", 4))
        pay_owed()
        log.info("queue: %s", status_line())
    except Exception:
        log.exception("poll crashed (continuing)")

def catchup():
    log.info("Startup catch-up...")
    poll()
    for _ in range(cfg["catchup_max_posts"]):
        if not publish(slot=False):
            break
        time.sleep(10)

def check_telegram():
    try:
        j = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getMe", timeout=20).json()
        if j.get("ok"):
            log.info("Telegram bot OK: @%s", j["result"]["username"])
        else:
            log.error("Bot token rejected: %s", j)
    except Exception as e:
        log.error("Cannot reach Telegram API (VPN/proxy on?): %s", str(e)[:120])

if __name__ == "__main__":
    log.info("=== started (%s mode) | models=%s ===", "TEST" if TEST else "NORMAL", MODELS)
    check_telegram()
    catchup()
    sch = BlockingScheduler(timezone=cfg["timezone"])
    if TEST:
        sch.add_job(publish, "interval", minutes=cfg["test_interval_minutes"],
                    coalesce=True, max_instances=1)
        poll_min = 1
        log.info("TEST: publishing every %d min, polling every 1 min",
                 cfg["test_interval_minutes"])
    else:
        for t in cfg["publish_times"]:
            h, m = t.split(":")
            sch.add_job(publish, "cron", hour=int(h), minute=int(m),
                        misfire_grace_time=3600, coalesce=True)
        poll_min = cfg["poll_minutes"]
        log.info("Publish times: %s | polling every %d min",
                 ", ".join(cfg["publish_times"]), poll_min)
    sch.add_job(poll, "interval", minutes=poll_min, max_instances=1, coalesce=True)
    try:
        sch.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("stopped")
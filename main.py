import hashlib, html, json, logging, os, re, socket, sqlite3, sys, threading, time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from logging.handlers import RotatingFileHandler

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from google import genai
from google.genai import types, errors as gerrors

load_dotenv()
cfg = json.load(open("config.json", encoding="utf-8"))

# ---------------- logging ----------------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%m-%d %H:%M:%S")
root = logging.getLogger()
root.setLevel(logging.INFO)
fh = RotatingFileHandler("bot.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
fh.setFormatter(fmt)
root.addHandler(fh)
if sys.stdout:
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
for n in ("httpx", "httpcore", "google_genai", "google_genai.models",
          "apscheduler", "urllib3"):
    logging.getLogger(n).setLevel(logging.WARNING)
log = logging.getLogger("bot")

# ---------------- single instance guard ----------------
_guard = socket.socket()
try:
    _guard.bind(("127.0.0.1", 47653))
except OSError:
    log.error("Another instance is already running. Exiting.")
    sys.exit(1)

if cfg.get("proxy"):
    os.environ["HTTP_PROXY"] = os.environ["HTTPS_PROXY"] = cfg["proxy"]

BOT_TOKEN = os.environ["BOT_TOKEN"]
ai = genai.Client(api_key=os.environ["GEMINI_KEY"])
MODELS = cfg["models"]
lock = threading.RLock()
UA = {"User-Agent": "Mozilla/5.0"}

# ---------------- database + memory ----------------
db = sqlite3.connect("queue.db", check_same_thread=False)
db.execute("""CREATE TABLE IF NOT EXISTS q(
  id INTEGER PRIMARY KEY AUTOINCREMENT, src TEXT, msg_id INTEGER,
  text TEXT, hash TEXT UNIQUE, ts REAL,
  status TEXT DEFAULT 'new',
  UNIQUE(src, msg_id))""")
db.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
for col, ddl in [("found_at", "REAL"), ("approved", "INTEGER DEFAULT 0"),
                 ("posted_at", "REAL"), ("method", "TEXT")]:
    try:
        db.execute(f"ALTER TABLE q ADD COLUMN {col} {ddl}")
    except sqlite3.OperationalError:
        pass
db.execute("UPDATE q SET found_at=ts WHERE found_at IS NULL")
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

def expire_old():
    cur = db.execute(
        "UPDATE q SET status='expired' WHERE status IN ('new','queued') AND ts<?",
        (time.time() - cfg["max_age_hours"] * 3600,))
    db.commit()
    if cur.rowcount:
        log.info("Expired %d old posts", cur.rowcount)

# ---------------- offline rule-based classifier ----------------
KEEP_WORDS = ["ثبت نام", "ثبتنام", "قرعه", "فروش", "بخشنامه", "اطلاعیه", "اطلاع رسانی",
              "افزایش قیمت", "کاهش قیمت", "قیمت جدید", "مهلت", "پیش فروش", "عرضه",
              "خبر فوری", "مصوبه", "تعطیل", "تمدید", "سهمیه", "یارانه", "وام",
              "آغاز شد", "اعلام شد", "شروع ثبت"]
AD_WORDS = ["تخفیف ویژه", "کد تخفیف", "عضو شوید", "جوین شوید", "فالو کنید",
            "لایک", "تبلیغ", "سفارش دهید", "پشتیبانی"]

def norm_fa(s):
    s = s.replace("ي", "ی").replace("ك", "ک").replace("\u200c", " ")
    return re.sub(r"\s+", " ", s)

LINK_RE = re.compile(r"https?://|t\.me/|\]\(|www\.")
EDU_PHRASES = ["چگونه", "نحوه ", "آموزش تصویری", "آموزش ثبت", "راهنمای ثبت", "راهنمای گام"]

def is_link_heavy(text):
    """True for link directories / posts whose value is the links, not the text."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    link_lines = sum(1 for l in lines if LINK_RE.search(l))
    n_links = len(re.findall(r"https?://|t\.me/", text))
    if n_links >= 3 and link_lines / len(lines) >= 0.5:
        return True
    body = re.sub(r"\s+", "", "".join(l for l in lines if not LINK_RE.search(l)))
    return len(body) < 40            # almost nothing left once links are removed

def drop_junk():
    """Reject link directories before they reach Gemini or the channel."""
    n = 0
    for rid, text in db.execute(
            "SELECT id,text FROM q WHERE status IN ('new','queued')").fetchall():
        if is_link_heavy(text):
            db.execute("UPDATE q SET status='rejected', approved=0, method='filter' "
                       "WHERE id=?", (rid,))
            log.info("REJECTED link-list/low-text post id=%s | %s", rid,
                     text.replace("\n", " ")[:50])
            n += 1
    db.commit()
    return n

def rule_classify(text):
    t = norm_fa(text)
    if is_link_heavy(text) or any(w in t for w in AD_WORDS + EDU_PHRASES):
        return False
    return len(t) >= 40 and any(w in t for w in KEEP_WORDS)

def rule_fallback():
    """Classify posts that waited too long for the AI, using keyword rules."""
    limit = time.time() - cfg["rule_after_minutes"] * 60
    rows = db.execute("SELECT id,text FROM q WHERE status='new' AND found_at<? "
                      "ORDER BY ts DESC", (limit,)).fetchall()
    if not rows:
        return
    kept = 0
    for rid, text in rows:
        ok = rule_classify(text)
        db.execute("UPDATE q SET status=?, approved=?, method='rule' WHERE id=?",
                   ("queued" if ok else "rejected", int(ok), rid))
        kept += ok
    db.commit()
    log.warning("AI unavailable -> rule-based fallback: %d posts -> %d queued, %d rejected",
                len(rows), kept, len(rows) - kept)

# ---------------- Gemini with fallback / backoff ----------------
PROMPT = """You are an editor for a Persian Telegram channel that publishes ONLY time-sensitive
news and registration announcements for the general reader. Posts below show links as
markdown: [label](url). For each post decide keep=true or keep=false.

keep=true ONLY if the post itself announces something specific and current, such as:
- an official circular/announcement, rule change, price change, important event, or
- a registration / sign-up / lottery announcement with its own details (dates, conditions, product).

keep=false if ANY of these is true:
- The post is mostly a list or directory of links (portals, websites, "links for X"),
  or more than about half of its lines are links. Its value is the links, not the text.
- It is educational or a guide: how-to, tutorial, tips, explanation of a process, general
  reference that is not tied to a specific new event.
- It has no concrete news/event/deadline of its own, and only points readers elsewhere.
- Ads, promos, greetings, chit-chat, polls, channel self-promotion, generic content.

Example to REJECT (keep=false): a post titled "link for registration of new university
students" followed by six lines of links to different university portals.
Example to KEEP (keep=true): a car-maker circular announcing a lottery sale with its
registration start and end dates.

When unsure, choose keep=false.
Return ONLY a JSON array with one entry per post: [{"id": <int>, "keep": true|false}]

Posts:
"""
blocked = {}
ai_pause_until = 0.0
fail_streak = 0

def parse_retry(msg):
    m = re.search(r"retry in ([\d.]+)s", msg, re.I) or \
        re.search(r"retryDelay'?\"?: ?'?\"?(\d+)s", msg)
    return float(m.group(1)) if m else None

def reset_ai_state():
    global ai_pause_until, fail_streak
    blocked.clear()
    ai_pause_until = 0.0
    fail_streak = 0

def call_ai(prompt):
    global ai_pause_until, fail_streak
    now = time.time()
    if now < ai_pause_until:
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
            if code == 400 and "location" in msg.lower():
                blocked[m] = now + 180
                log.warning("[%s] Gemini: location not supported -> VPN off, or Google "
                            "traffic bypasses VPN, or exit country unsupported. Retry in 3 min",
                            m)
            elif code == 429:
                per_day = "PerDay" in msg or "per day" in msg.lower()
                wait = 3 * 3600 if per_day else (parse_retry(msg) or 60) + 5
                blocked[m] = now + wait
                log.warning("[%s] quota limit (429) -> skipping until %s", m, hhmm(blocked[m]))
            elif code in (500, 502, 503, 504):
                wait = min(120 * (2 ** fail_streak), 1800)
                fail_streak += 1
                blocked[m] = now + wait
                log.warning("[%s] Google overloaded (%s) -> skipping until %s",
                            m, code, hhmm(blocked[m]))
            else:
                blocked[m] = now + 3600
                log.error("[%s] API error %s: %s -> skipping 1h (check model name/key)",
                          m, code, msg[:150])
        except Exception as e:
            ai_pause_until = now + 120
            log.error("Cannot reach Gemini: %s", str(e)[:150])
            return None, None
    return None, None

def classify_pending():
    """Returns None if AI unavailable, 0 if nothing to do, else number decided."""
    rows = db.execute("SELECT id,text FROM q WHERE status='new' ORDER BY ts DESC LIMIT ?",
                      (cfg["batch_size"],)).fetchall()
    if not rows:
        return 0
    payload = "\n".join(f"### ID {rid}\n{text[:700]}\n" for rid, text in rows)
    out, model = call_ai(PROMPT + payload)
    if out is None:
        return None
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
            db.execute("UPDATE q SET status=?, approved=?, method='ai' WHERE id=?",
                       ("queued" if ok else "rejected", int(ok), rid))
            kept += ok
            rejected += (not ok)
        db.commit()
        log.info("AI [%s]: %d posts -> %d queued, %d rejected", model,
                 kept + rejected, kept, rejected)
        return kept + rejected
    except Exception as e:
        log.error("Bad AI response, will retry later: %s | %s", e, out[:150])
        return None

# ---------------- scraping with memory ----------------
def norm_hash(text):
    return hashlib.md5(re.sub(r"\W+", "", text).encode()).hexdigest()

def parse_page(page_html):
    soup = BeautifulSoup(page_html, "html.parser")
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
    r = requests.get(url, timeout=15, headers=UA)
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
                        db.execute("INSERT INTO q(src,msg_id,text,hash,ts,found_at) "
                                   "VALUES(?,?,?,?,?,?)",
                                   (ch, it["id"], t, norm_hash(t), it["ts"], time.time()))
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
                log.warning("%s: more posts than max_pages covers; older ones skipped", ch)
            log.info("crawl %-18s last_id=%s | new=%d dup=%d", ch, top, added, dups)
        except requests.RequestException as e:
            log.error("crawl %s failed: %s", ch, str(e)[:100])
        except Exception:
            log.exception("crawl %s crashed", ch)

# ---------------- editing + publishing ----------------
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

def publish_row(row, tag=""):
    rid, text = row
    body = clean(text)
    if body.strip() == cfg["footer"]:
        db.execute("UPDATE q SET status='rejected' WHERE id=?", (rid,))
        db.commit()
        return False
    try:
        send(body)
    except Exception as e:
        log.error("Send failed (will retry next cycle): %s", str(e)[:150])
        return False
    now = time.time()
    db.execute("UPDATE q SET status='posted', posted_at=? WHERE id=?", (now, rid))
    db.commit()
    meta_set("last_posted_ts", now)
    log.info("POSTED%s id=%s | %s", tag, rid, body.replace("\n", " ")[:60])
    return True

# ---------------- manual-post / duplicate guard ----------------
db.execute("CREATE TABLE IF NOT EXISTS mine(msg_id INTEGER PRIMARY KEY, ts REAL, text TEXT)")
db.commit()
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_mine_sync = 0.0
_mine_ts = []

def body_of(text):
    """Same cleaning as clean() but without the footer: drop link/mention lines."""
    return "\n".join(l for l in text.splitlines()
                     if not re.search(r"https?://|t\.me/|\]\(|@\w+", l))

def sig(text):
    t = norm_fa(body_of(text)).translate(DIGITS).lower()
    words = set(re.findall(r"\w+", t))
    return words, {w for w in words if w.isdigit()}

def similar(a, b):
    (wa, na), (wb, nb) = a, b
    if not wa or not wb:
        return False
    if na and nb and not (na <= nb or nb <= na):   # different dates/prices => different news
        return False
    inter = len(wa & wb)
    jac = inter / len(wa | wb)
    cont = inter / min(len(wa), len(wb))
    return jac >= cfg.get("dup_similarity", 0.7) or (cont >= 0.9 and min(len(wa), len(wb)) >= 8)

def sync_mine(force=False):
    """Read YOUR channel page and remember every post, manual ones included.
    Returns timestamps of recent posts, or None if the page can't be read."""
    global _mine_sync, _mine_ts
    if not force and time.time() - _mine_sync < 60:
        return _mine_ts
    try:
        items = fetch_page(cfg["target"].lstrip("@"))
    except Exception as e:
        log.warning("Could not read your channel page, using local record: %s", str(e)[:80])
        return None
    for it in items:
        if it["text"]:
            db.execute("INSERT OR REPLACE INTO mine(msg_id,ts,text) VALUES(?,?,?)",
                       (it["id"], it["ts"], it["text"]))
    db.execute("DELETE FROM mine WHERE ts<?",
               (time.time() - cfg.get("dup_lookback_hours", 72) * 3600,))
    db.commit()
    _mine_sync = time.time()
    _mine_ts = [i["ts"] for i in items]
    return _mine_ts

def channel_recent():
    return sync_mine()

def refs():
    """Signatures of everything already in your channel (manual + bot posts)."""
    since = time.time() - cfg.get("dup_lookback_hours", 72) * 3600
    rows = db.execute("SELECT text FROM mine WHERE ts>?", (since,)).fetchall()
    rows += db.execute("SELECT text FROM q WHERE status='posted' AND posted_at>?",
                       (since,)).fetchall()
    return [sig(t) for (t,) in rows if t]

def is_dup(text, R):
    s = sig(text)
    return any(similar(s, r) for r in R)

def drop_duplicates():
    """Mark waiting posts that your channel already has (e.g. you posted it by hand)."""
    R = refs()
    if not R:
        return 0
    n = 0
    for rid, text in db.execute(
            "SELECT id,text FROM q WHERE status IN ('new','queued')").fetchall():
        if is_dup(text, R):
            db.execute("UPDATE q SET status='duplicate' WHERE id=?", (rid,))
            log.info("DUPLICATE of a post already in your channel -> skipped id=%s | %s",
                     rid, body_of(text).replace("\n", " ")[:50])
            n += 1
    db.commit()
    return n

PACE_DEFAULT = {"drain_hours": 6, "keepup": 0.7, "min_gap_minutes": 20,
                "max_per_hour": 3, "hard_cap_per_day": 16, "rate_window_hours": 3}

def pace_cfg():
    p = dict(PACE_DEFAULT)
    p.update(cfg.get("pace", {}))
    return p

def compute_pace(queued, now, hour_now):
    """Smart pacing: gap shrinks with backlog AND with the arrival rate of approved posts.
    desired posts/hour = backlog / drain_time + arrival_rate * keepup."""
    p = pace_cfg()
    rw = p["rate_window_hours"]
    arrived = db.execute("SELECT COUNT(*) FROM q WHERE approved=1 AND ts>?",
                         (now - rw * 3600,)).fetchone()[0]
    rate = arrived / rw                                  # approved posts per hour
    left = max(1.0, cfg["window_end"] - hour_now)        # posting hours left today
    drain = max(1.0, min(p["drain_hours"], left))
    want = queued / drain + rate * p["keepup"]           # posts per hour we should publish
    if want <= 0:
        gap = cfg["gap_normal_minutes"]
    else:
        gap = 60.0 / want
    gap = max(p["min_gap_minutes"], min(cfg["gap_normal_minutes"], gap))
    cap = cfg["max_posts_per_day"]
    if want > 0 and gap < cfg["gap_normal_minutes"]:     # busy: allow a higher daily cap
        cap = max(cap, p["hard_cap_per_day"])
    return gap, cap, rate, want

def smart_publish(tg_up):
    with lock:
        expire_old()
        queued = db.execute("SELECT COUNT(*) FROM q WHERE status='queued'").fetchone()[0]
        if not tg_up:
            if queued:
                log.warning("%d post(s) ready but Telegram unreachable. "
                            "Will post when VPN is back.", queued)
            return
        now = time.time()
        h = datetime.now().hour
        if not (cfg["window_start"] <= h < cfg["window_end"]):
            log.info("Outside posting window (%d-%d); queued=%d",
                     cfg["window_start"], cfg["window_end"], queued)
            return
        since = now - 86400
        count = db.execute("SELECT COUNT(*) FROM q WHERE status='posted' AND posted_at>?",
                           (since,)).fetchone()[0]
        last = float(meta_get("last_posted_ts", 0))
        ts_list = channel_recent()
        if ts_list is not None:
            count = max(count, len([t for t in ts_list if t > since]))
            if ts_list:
                last = max(last, max(ts_list))
        need, cap, rate, want = compute_pace(queued, now, datetime.now().hour + datetime.now().minute / 60)
        if count >= cap:
            log.info("24h cap reached (%d/%d) | queued=%d", count, cap, queued)
            return
        if count < cfg["min_posts_per_day"]:
            need = min(need, cfg["gap_below_min_minutes"])
        last_hour = db.execute("SELECT COUNT(*) FROM q WHERE status='posted' AND posted_at>?",
                               (now - 3600,)).fetchone()[0]
        if last_hour >= pace_cfg()["max_per_hour"]:
            log.info("Hourly cap reached (%d) | queued=%d", last_hour, queued)
            return
        gap = (now - last) / 60 if last else 10 ** 6
        if gap < need:
            log.info("24h=%d/%d | queued=%d arrivals=%.1f/h -> target gap %.0f min | "
                     "last post %.0f min ago, next in %.0f min",
                     count, cap, queued, rate, need, gap, need - gap)
            return
        sync_mine(force=True)          # fresh look at your channel right before posting
        drop_junk()
        drop_duplicates()
        R = refs()
        row, tag = None, ""
        for r in db.execute("SELECT id,text FROM q WHERE status='queued' "
                            "ORDER BY ts DESC LIMIT 15").fetchall():
            if is_dup(r[1], R):
                db.execute("UPDATE q SET status='duplicate' WHERE id=?", (r[0],))
                continue
            row = r
            break
        if not row and count < cfg["min_posts_per_day"] and h >= cfg["fallback_from_hour"]:
            for r in db.execute(
                    "SELECT id,text FROM q WHERE approved=1 AND status='expired' AND ts>? "
                    "ORDER BY ts DESC LIMIT 15",
                    (now - cfg["fallback_max_age_hours"] * 3600,)).fetchall():
                if is_link_heavy(r[1]) or is_dup(r[1], R):
                    db.execute("UPDATE q SET status='duplicate' WHERE id=?", (r[0],))
                    continue
                row, tag = r, " (fallback: older approved item to reach daily minimum)"
                break
        db.commit()
        if not row:
            log.info("24h count=%d | nothing ready to post (queue empty)", count)
            return
        log.info("24h count=%d, gap ok -> posting newest", count)
        publish_row(row, tag)

# ---------------- main cycle ----------------
was_up = None
bot_checked = False

def reachable(url, t=8):
    try:
        requests.get(url, timeout=t, headers=UA)
        return True
    except Exception:
        return False

def check_telegram():
    try:
        j = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getMe", timeout=15).json()
        if j.get("ok"):
            log.info("Telegram bot OK: @%s", j["result"]["username"])
        else:
            log.error("Bot token rejected: %s", j)
    except Exception as e:
        log.error("Cannot reach Telegram API: %s", str(e)[:100])

def poll():
    global was_up, bot_checked
    try:
        tg_up = reachable("https://t.me")
        if tg_up and not was_up:
            log.info("Connection to Telegram is UP (VPN on) -> resetting AI state")
            reset_ai_state()
        if not tg_up and was_up is not False:
            log.warning("Telegram unreachable (VPN off?) -> offline mode: "
                        "no crawl/post, existing queue kept")
        was_up = tg_up
        if tg_up and not bot_checked:
            check_telegram()
            bot_checked = True
        with lock:
            expire_old()
            if tg_up:
                sync_mine()
                collect()
                drop_junk()            # link directories: no AI needed
                drop_duplicates()      # before AI: saves quota
            ai_ok = True
            for _ in range(10):
                r = classify_pending()
                if r is None:
                    ai_ok = False
                    break
                if r == 0:
                    break
                time.sleep(cfg["ai_gap_seconds"])
            if not ai_ok:
                rule_fallback()
        smart_publish(tg_up)
        log.info("queue: %s", status_line())
    except Exception:
        log.exception("poll crashed (continuing)")

class Dash(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            lines = open("bot.log", encoding="utf-8").read().splitlines()[-150:]
        except Exception:
            lines = []
        try:
            c = sqlite3.connect("queue.db")
            rows = c.execute("SELECT status, COUNT(*) FROM q GROUP BY status").fetchall()
            c.close()
            stat = " | ".join(f"{s}={n}" for s, n in rows) or "empty"
        except Exception:
            stat = "db busy"
        body = (
            '<meta charset="utf-8"><meta http-equiv="refresh" content="10">'
            '<body style="background:#111;color:#ddd;font-family:Consolas,monospace">'
            f"<h3>{html.escape(stat)}</h3>"
            f'<pre style="white-space:pre-wrap">{html.escape(chr(10).join(reversed(lines)))}</pre>'
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, *a):
        pass

def start_dashboard():
    port = cfg.get("dashboard_port")
    if not port:
        return
    try:
        srv = HTTPServer(("127.0.0.1", int(port)), Dash)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        log.info("Dashboard: http://127.0.0.1:%s", port)
    except OSError as e:
        log.warning("Dashboard not started: %s", e)

if __name__ == "__main__":
    log.info("=== started | posts/24h: %d-%d | window=%d-%d | max age=%dh ===",
             cfg["min_posts_per_day"], cfg["max_posts_per_day"],
             cfg["window_start"], cfg["window_end"], cfg["max_age_hours"])
    start_dashboard()
    poll()
    sch = BlockingScheduler()
    sch.add_job(poll, "interval", minutes=cfg["poll_minutes"],
                max_instances=1, coalesce=True)
    try:
        sch.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("stopped")
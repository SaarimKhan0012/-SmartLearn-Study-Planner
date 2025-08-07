# ---------- Auto-install missing packages ----------
import subprocess, sys
for _pkg in ["streamlit", "pandas", "matplotlib", "requests", "beautifulsoup4"]:
    try:
        __import__("bs4" if _pkg == "beautifulsoup4" else _pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg])

# ---------- Imports (after auto-install) ----------
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import math, random
from datetime import date, datetime, timedelta, time as dtime

# ML (PyTorch) — not auto-installed to keep first-run lighter; if missing, we’ll degrade gracefully.
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except Exception:
    TORCH_OK = False

# Charts
import matplotlib.pyplot as plt

# Web fetch
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# ================== CONFIG ==================
DB_PATH = "smartlearn.db"
SESSIONS_PER_DAY = 3
SESSION_BLOCK_MIN = [25, 25, 25]
START_TIMES = [dtime(9,0), dtime(10,0), dtime(16,0)]

random.seed(42); np.random.seed(42)

# ================== OPTIONAL BUILT-IN SUBJECTS ==================
# You can pre-seed common subjects here. Topics can be empty; you’ll fetch them from the web or add in UI.
BUILTIN_SUBJECTS = [
    # PUC Science
    "PUC Physics", "PUC Chemistry", "PUC Mathematics", "PUC Biology",
    "PUC Computer Science", "PUC Electronics", "PUC Statistics",
    # PUC Commerce/Arts
    "PUC Accountancy", "PUC Business Studies", "PUC Economics",
    "PUC History", "PUC Geography", "PUC Political Science", "PUC Sociology",
    # Engineering / Higher Ed
    "CSE (Engineering)", "ECE (Engineering)", "Mechanical (Engineering)", "Civil (Engineering)",
    "AI/ML (Engineering)", "EEE (Engineering)"
]

# ================== DB LAYER ==================
def db():
    return sqlite3.connect(DB_PATH, check_same_thread=False, timeout=5.0)

def init_db():
    conn = db(); cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        name TEXT
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS subjects(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS topics(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id INTEGER NOT NULL,
        topic TEXT NOT NULL,
        difficulty INTEGER NOT NULL CHECK(difficulty BETWEEN 1 AND 5),
        UNIQUE(subject_id, topic),
        FOREIGN KEY(subject_id) REFERENCES subjects(id)
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS quiz_events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        topic_id INTEGER NOT NULL,
        ts TEXT NOT NULL,
        correct INTEGER NOT NULL,
        time_sec REAL
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS bandit_memory(
        user_id INTEGER NOT NULL,
        topic_id INTEGER NOT NULL,
        reward REAL NOT NULL,
        ts TEXT NOT NULL
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS study_sessions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        subject_id INTEGER NOT NULL,
        sess_date TEXT NOT NULL,
        sess_time TEXT NOT NULL,
        topic_id INTEGER NOT NULL,
        duration_min INTEGER NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('pending','done','skipped')),
        created_at TEXT NOT NULL
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS manual_schedule(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        subject_id INTEGER NOT NULL,
        dow INTEGER NOT NULL CHECK(dow BETWEEN 0 AND 6),
        start_time TEXT NOT NULL,
        end_time TEXT NOT NULL,
        topic_id INTEGER,
        created_at TEXT NOT NULL
    );""")
    conn.commit(); conn.close()
    seed_subjects()

def seed_subjects():
    conn=db(); cur=conn.cursor()
    for s in BUILTIN_SUBJECTS:
        cur.execute("INSERT OR IGNORE INTO subjects(name) VALUES(?)", (s,))
    conn.commit(); conn.close()

# ---- users
def get_user(email):
    conn=db(); cur=conn.cursor()
    cur.execute("SELECT id,email,name FROM users WHERE email=?", (email,))
    row=cur.fetchone(); conn.close(); return row

def create_user(email, name):
    conn=db(); cur=conn.cursor()
    cur.execute("INSERT OR IGNORE INTO users(email,name) VALUES(?,?)",(email,name))
    conn.commit(); conn.close()
    return get_user(email)

# ---- subjects/topics
def list_subjects():
    conn=db(); cur=conn.cursor()
    cur.execute("SELECT id,name FROM subjects ORDER BY name")
    rows=cur.fetchall(); conn.close(); return rows

def subject_id_by_name(name: str):
    conn=db(); cur=conn.cursor()
    cur.execute("SELECT id FROM subjects WHERE name=?", (name,))
    row=cur.fetchone(); conn.close()
    return row[0] if row else None

def ensure_subject(name: str):
    conn=db(); cur=conn.cursor()
    cur.execute("INSERT OR IGNORE INTO subjects(name) VALUES(?)", (name,))
    conn.commit()
    cur.execute("SELECT id FROM subjects WHERE name=?", (name,))
    sid = cur.fetchone()[0]
    conn.close(); return sid

def get_topics(subject_id: int):
    conn=db(); cur=conn.cursor()
    cur.execute("SELECT id, topic, difficulty FROM topics WHERE subject_id=? ORDER BY difficulty, topic", (subject_id,))
    rows=cur.fetchall(); conn.close(); return rows

def add_topic(subject_id: int, topic: str, difficulty: int):
    conn=db(); cur=conn.cursor()
    cur.execute("INSERT OR IGNORE INTO topics(subject_id, topic, difficulty) VALUES(?,?,?)",
                (subject_id, topic, int(difficulty)))
    conn.commit(); conn.close()

def update_topic(topic_id: int, topic: str, difficulty: int):
    conn=db(); cur=conn.cursor()
    cur.execute("UPDATE topics SET topic=?, difficulty=? WHERE id=?", (topic, int(difficulty), int(topic_id)))
    conn.commit(); conn.close()

def delete_topic(topic_id: int):
    conn=db(); cur=conn.cursor()
    cur.execute("DELETE FROM topics WHERE id=?", (int(topic_id),))
    conn.commit(); conn.close()

# ---- events & bandit
def get_quiz_events(uid, subject_id=None):
    conn=db(); cur=conn.cursor()
    if subject_id is None:
        cur.execute("SELECT topic_id, ts, correct, time_sec FROM quiz_events WHERE user_id=? ORDER BY ts ASC",(uid,))
    else:
        cur.execute("""SELECT q.topic_id, q.ts, q.correct, q.time_sec
                       FROM quiz_events q JOIN topics t ON q.topic_id=t.id
                       WHERE q.user_id=? AND t.subject_id=?
                       ORDER BY q.ts ASC""",(uid,subject_id))
    rows=cur.fetchall(); conn.close(); return rows

def insert_quiz(uid, topic_id, correct, time_sec):
    conn=db(); cur=conn.cursor()
    cur.execute("""INSERT INTO quiz_events(user_id,topic_id,ts,correct,time_sec)
                   VALUES(?,?,?,?,?)""",(uid, topic_id, datetime.now().isoformat(), int(correct), float(time_sec)))
    conn.commit(); conn.close()

def insert_bandit(uid, topic_id, reward):
    conn=db(); cur=conn.cursor()
    cur.execute("""INSERT INTO bandit_memory(user_id,topic_id,reward,ts)
                   VALUES(?,?,?,?)""",(uid, topic_id, float(reward), datetime.now().isoformat()))
    conn.commit(); conn.close()

def bandit_stats(uid, subject_id=None):
    conn=db(); cur=conn.cursor()
    if subject_id is None:
        cur.execute("""SELECT topic_id, AVG(reward), COUNT(*)
                       FROM bandit_memory WHERE user_id=? GROUP BY topic_id""",(uid,))
    else:
        cur.execute("""SELECT b.topic_id, AVG(b.reward), COUNT(*)
                       FROM bandit_memory b JOIN topics t ON b.topic_id=t.id
                       WHERE b.user_id=? AND t.subject_id=? GROUP BY b.topic_id""",(uid,subject_id))
    rows=cur.fetchall(); conn.close()
    return {tid: {"avg": float(avg), "n": int(n)} for tid,avg,n in rows}

# ---- sessions
def sessions_for_week(uid, subject_id, monday: date, sunday: date):
    conn=db(); cur=conn.cursor()
    cur.execute("""SELECT id, sess_date, sess_time, topic_id, duration_min, status
                   FROM study_sessions
                   WHERE user_id=? AND subject_id=? AND sess_date BETWEEN ? AND ?
                   ORDER BY sess_date, sess_time""",
                (uid, subject_id, monday.isoformat(), sunday.isoformat()))
    rows=cur.fetchall(); conn.close(); return rows

def upsert_session(uid, subject_id, sess_date: date, sess_time: dtime, topic_id: int, duration_min: int, status="pending"):
    conn=db(); cur=conn.cursor()
    cur.execute("""INSERT INTO study_sessions(user_id, subject_id, sess_date, sess_time, topic_id, duration_min, status, created_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (uid, subject_id, sess_date.isoformat(), sess_time.strftime("%H:%M"), topic_id, duration_min, status, datetime.now().isoformat()))
    conn.commit(); conn.close()

def update_session_status(session_id: int, status: str):
    conn=db(); cur=conn.cursor()
    cur.execute("UPDATE study_sessions SET status=? WHERE id=?", (status, int(session_id)))
    conn.commit(); conn.close()

# ---- manual schedule
def list_manual_slots(uid: int, subject_id: int):
    conn = db(); cur = conn.cursor()
    cur.execute("""SELECT id, dow, start_time, end_time, topic_id
                   FROM manual_schedule
                   WHERE user_id=? AND subject_id=?
                   ORDER BY dow, start_time""", (uid, subject_id))
    rows = cur.fetchall(); conn.close(); return rows

def add_manual_slot(uid: int, subject_id: int, dow: int, start_time: str, end_time: str, topic_id: int | None):
    conn = db(); cur = conn.cursor()
    cur.execute("""INSERT INTO manual_schedule(user_id,subject_id,dow,start_time,end_time,topic_id,created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (uid, subject_id, dow, start_time, end_time, topic_id, datetime.now().isoformat()))
    conn.commit(); conn.close()

def delete_manual_slot(slot_id: int):
    conn = db(); cur = conn.cursor()
    cur.execute("DELETE FROM manual_schedule WHERE id=?", (int(slot_id),))
    conn.commit(); conn.close()

# ================== ML (KT LSTM) ==================
# If torch is unavailable, the app falls back to a simple heuristic mastery.
if TORCH_OK:
    class KTLSTM(nn.Module):
        def __init__(self, num_topics: int, hidden=64, emb=16):
            super().__init__()
            self.emb = nn.Embedding(num_topics, emb)
            self.lstm = nn.LSTM(input_size=emb+1, hidden_size=hidden, batch_first=True)
            self.out = nn.Linear(hidden, 1)
        def forward(self, topic_ids, correctness):
            x = self.emb(topic_ids)
            x = torch.cat([x, correctness], dim=-1)
            h,_ = self.lstm(x)
            return self.out(h)

def build_seq(events, idx_map):
    if not events: return None
    tids = [idx_map[e[0]] for e in events]
    corr = [float(e[2]) for e in events]
    tid = np.array(tids, dtype=np.int64)[None, :]
    cor = np.array(corr, dtype=np.float32)[None, :, None]
    return tid, cor

def train_and_estimate_mastery(topics, events, idx_map):
    # Fallback neutral mastery
    mastery = {tid: 0.5 for tid,_,_ in topics}
    if not events:
        return mastery
    if TORCH_OK:
        model = KTLSTM(num_topics=len(topics))
        tid, cor = build_seq(events, idx_map)
        tid_t = torch.tensor(tid, dtype=torch.long)
        cor_t = torch.tensor(cor, dtype=torch.float32)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        loss = nn.BCEWithLogitsLoss()
        model.train()
        for _ in range(4):
            logits = model(tid_t, cor_t)
            l = loss(logits, cor_t)
            opt.zero_grad(); l.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(tid_t, cor_t)).numpy()[0,:,0]
        last = {}
        for e,p in zip(events, probs): last[e[0]] = float(p)
        for t in mastery: mastery[t] = last.get(t, 0.5)
        return mastery
    else:
        # Simple rolling accuracy by topic as a proxy
        df = pd.DataFrame(events, columns=["topic_id","ts","correct","time_sec"])
        for tid in df["topic_id"].unique():
            sub = df[df["topic_id"]==tid].sort_values("ts")
            mastery[tid] = float(sub["correct"].rolling(3, min_periods=1).mean().iloc[-1])
        return mastery

# ================== Recs (bandit + mastery/recency) ==================
def reward(correct: int, time_sec: float):
    base = 1.0 if correct else 0.0
    bonus = max(min((60.0 - float(time_sec))/60.0, 1.0), 0.0) * 0.3
    return base + bonus

def recency_component(last_seen_map, tid):
    if tid not in last_seen_map: return 1.0
    days = max((datetime.now() - last_seen_map[tid]).days, 1)
    return 1.0 / math.sqrt(days)

def recommend(topics, mastery_map, last_seen_map, bandit, k=3, eps=0.2):
    W_M, W_R, W_B = 0.5, 0.25, 0.25
    MASTERED = 0.85
    cands = [(tid,t,df) for tid,t,df in topics if mastery_map.get(tid,0.5) < MASTERED] or topics
    scored=[]
    for tid,t,df in cands:
        m = mastery_map.get(tid,0.5)
        r = recency_component(last_seen_map, tid)
        b = bandit.get(tid, {"avg":0.0})["avg"]
        s = W_M*(1-m) + W_R*r + W_B*b
        scored.append((s, tid, t, df))
    scored.sort(reverse=True)
    picks=[]; pool=scored.copy()
    while len(picks) < min(k,len(pool)):
        choice = random.choice(pool[len(pool)//2:] if random.random()<eps and len(pool)>1 else pool[:1])
        picks.append((choice[1], choice[2], choice[3]))
        pool.remove(choice)
    return picks

# ================== Planner generation ==================
def monday_of_week(dt: date):
    return dt - timedelta(days=dt.weekday())

def generate_week_plan(uid, subject_id, topics, mastery_map, last_seen_map, bandit, planner_mode: str):
    today = date.today()
    mon = monday_of_week(today); sun = mon + timedelta(days=6)
    existing = sessions_for_week(uid, subject_id, mon, sun)
    if existing: return mon, sun

    manual_slots = list_manual_slots(uid, subject_id)
    if planner_mode == "Manual (my own times)" and manual_slots:
        base_recs = recommend(topics, mastery_map, last_seen_map, bandit, k=10, eps=0.2)
        queue = [r[0] for r in base_recs] * 3; q_idx = 0
        for d in range(7):
            day_slots = [s for s in manual_slots if s[1] == d]   # (id, dow, start, end, topic_id)
            day_slots.sort(key=lambda r: r[2])
            sess_date = mon + timedelta(days=d)
            for _, _, start_str, end_str, pinned_tid in day_slots:
                t0 = datetime.strptime(start_str, "%H:%M"); t1 = datetime.strptime(end_str, "%H:%M")
                duration_min = int((t1 - t0).total_seconds() / 60)
                if duration_min <= 0: continue
                chosen_tid = pinned_tid
                if not chosen_tid:
                    if q_idx >= len(queue):
                        base_recs = recommend(topics, mastery_map, last_seen_map, bandit, k=10, eps=0.2)
                        queue = [r[0] for r in base_recs] * 2; q_idx = 0
                    chosen_tid = queue[q_idx]; q_idx += 1
                upsert_session(uid, subject_id, sess_date, t0.time(), chosen_tid, duration_min, status="pending")
        return mon, sun

    # Auto mode fallback
    base_recs = recommend(topics, mastery_map, last_seen_map, bandit, k=5, eps=0.2)
    queue = [r[0] for r in base_recs] * 3; idx = 0
    for d in range(7):
        sess_date = mon + timedelta(days=d)
        for slot_i in range(min(SESSIONS_PER_DAY, len(START_TIMES))):
            if idx >= len(queue): break
            upsert_session(uid, subject_id, sess_date, START_TIMES[slot_i], queue[idx],
                           SESSION_BLOCK_MIN[slot_i % len(SESSION_BLOCK_MIN)], status="pending")
            idx += 1
        if idx >= len(queue): break
    return mon, sun

# ================== SYLLABUS FETCH ==================
def parse_text_items_as_topics(html: str):
    """Generic fallback: extract <li> and <h2>/<h3> as topic lines."""
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for li in soup.select("li"):
        txt = li.get_text(" ", strip=True)
        if 3 <= len(txt) <= 140:
            items.append(txt)
    for h in soup.select("h2, h3"):
        txt = h.get_text(" ", strip=True)
        if 3 <= len(txt) <= 140:
            items.append(txt)
    # dedupe keeping order
    seen = set(); out=[]
    for t in items:
        low = t.lower()
        if low in seen: continue
        seen.add(low); out.append(t)
    return out[:200]

def karnataka_puc_biology_known_urls():
    return [
        "https://school.careers360.com/boards/dpue-karnataka/karnataka-board-2nd-puc-biology-syllabus-bsar",
        "https://www.shiksha.com/boards/karnataka-2nd-puc-board-syllabus"
    ]

def fetch_url_text(url: str, timeout=12):
    headers = {"User-Agent": "Mozilla/5.0 (smartlearn-app)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text

def guess_difficulty(line: str):
    l = len(line)
    if l < 20: return 2
    if l < 40: return 3
    if l < 70: return 4
    return 5

def import_topics_to_subject(subject_id: int, topic_lines: list[str]):
    for t in topic_lines:
        add_topic(subject_id, t, guess_difficulty(t))

def fetch_syllabus_for_subject(subject_name: str):
    """
    Returns (topics:list[str], source_url:str|None, note:str)
    - Known handler: 'Biology 2nd PUC' / 'PUC Biology'
    - Otherwise returns empty (user can paste a URL)
    """
    s = subject_name.lower()
    if "biology" in s and ("2nd puc" in s or "puc" in s or "class 12" in s):
        for u in karnataka_puc_biology_known_urls():
            try:
                html = fetch_url_text(u)
                items = parse_text_items_as_topics(html)
                filtered = [x for x in items if len(x.split())<=12 and not x.endswith(":")]
                if len(filtered) < 8:
                    filtered = items[:40]
                if filtered:
                    return filtered, u, "Fetched from a known syllabus page."
            except Exception:
                continue
        return [], None, "Tried known sources but couldn’t parse."
    return [], None, "No built-in fetcher for this subject. Paste a URL below."

# ================== UI ==================
st.set_page_config(page_title="SmartLearn Planner", page_icon="📅", layout="wide")
st.title("📅 SmartLearn — Study Planner")

init_db()

# --- Sidebar: auth ---
with st.sidebar:
    st.header("Account")
    email = st.text_input("Email", value="you@example.com")
    name = st.text_input("Name", value="Student")
    if st.button("Sign in / Create"):
        user = create_user(email, name)
        st.session_state["uid"] = user[0]
        st.success(f"Signed in as {email}")

uid = st.session_state.get("uid")
if not uid:
    st.info("Sign in to continue.")
    st.stop()

# --- Sidebar: subjects & topics ---
with st.sidebar:
    st.header("Subjects & Topics")

    # quick add subject
    new_quick = st.text_input("Add new subject ")
    if st.button("Add Subject"):
        if new_quick.strip():
            ensure_subject(new_quick.strip())
            st.success("Subject added.")
            st.rerun()

    subs = list_subjects()
    sub_names = [s[1] for s in subs]
    if not sub_names:
        st.error("No subjects available. Add one above.")
        st.stop()

    chosen_name = st.selectbox("Choose Subject", sub_names)
    subject_id = subject_id_by_name(chosen_name)

    # === Syllabus fetch ===
    st.subheader("Fetch Syllabus from Web")
    colfs1, colfs2 = st.columns([1,1])
    with colfs1:
        if st.button("Auto-Fetch (e.g., PUC Biology)"):
            topics, src, note = fetch_syllabus_for_subject(chosen_name)
            if topics:
                import_topics_to_subject(subject_id, topics)
                st.success(f"Imported {len(topics)} topics. {note}")
                if src: st.caption(f"Source: {urlparse(src).netloc}")
                st.rerun()
            else:
                st.warning(f"No topics imported. {note}")
    with colfs2:
        custom_url = st.text_input("Or paste a syllabus URL")
        if st.button("Fetch from URL"):
            if not custom_url.strip():
                st.warning("Enter a URL first.")
            else:
                try:
                    html = fetch_url_text(custom_url.strip())
                    items = parse_text_items_as_topics(html)
                    filtered = [x for x in items if len(x.split())<=14 and not x.endswith(":")]
                    topics = filtered or items[:50]
                    if topics:
                        import_topics_to_subject(subject_id, topics)
                        st.success(f"Imported {len(topics)} topics from URL.")
                        st.caption(f"Source: {urlparse(custom_url).netloc}")
                        st.rerun()
                    else:
                        st.warning("Couldn’t parse topics from that page.")
                except Exception as e:
                    st.error(f"Fetch error: {e}")

    with st.expander("📝 Manage Topics"):
        topics_df = pd.DataFrame(get_topics(subject_id), columns=["id","topic","difficulty"])
        st.dataframe(topics_df, use_container_width=True, hide_index=True)
        st.subheader("Add / Edit Topic")
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            t_name = st.text_input("Topic name")
        with col2:
            t_diff = st.number_input("Difficulty (1-5)", min_value=1, max_value=5, value=2, step=1)
        with col3:
            st.write("")
            if st.button("Add Topic"):
                if t_name.strip():
                    add_topic(subject_id, t_name.strip(), int(t_diff))
                    st.success("Topic added.")
                    st.rerun()
                else:
                    st.warning("Enter a topic name.")
        if not topics_df.empty:
            edit_row = st.selectbox("Select a topic to edit/delete (ID)", topics_df["id"].tolist())
            sel = topics_df[topics_df["id"]==edit_row].iloc[0]
            e_name = st.text_input("Edit name", value=sel["topic"])
            e_diff = st.number_input("Edit difficulty", min_value=1, max_value=5, value=int(sel["difficulty"]), step=1, key="ediff")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Save Changes"):
                    update_topic(int(edit_row), e_name.strip(), int(e_diff))
                    st.success("Updated.")
                    st.rerun()
            with c2:
                if st.button("Delete Topic"):
                    delete_topic(int(edit_row))
                    st.success("Deleted.")
                    st.rerun()

    # Planner Mode
    st.header("Planner Mode")
    planner_mode = st.radio("How to schedule this subject:", ["Auto (AI)", "Manual (my own times)"], index=0)

    # Manual schedule editor
    topics_for_select = get_topics(subject_id)
    if planner_mode == "Manual (my own times)":
        st.subheader("My time slots")
        ms = list_manual_slots(uid, subject_id)
        if ms:
            slot_rows = []
            for sid, dow, s, e, tid in ms:
                slot_rows.append({"ID": sid, "Day": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][dow],
                                  "Start": s, "End": e, "Topic ID (opt.)": tid})
            st.dataframe(pd.DataFrame(slot_rows), use_container_width=True, hide_index=True)
            del_id = st.number_input("Delete slot by ID", min_value=0, value=0, step=1)
            if del_id and st.button("Delete Slot"):
                delete_manual_slot(int(del_id))
                st.success("Deleted.")
                st.rerun()
        else:
            st.caption("No manual slots yet.")
        st.markdown("**Add a slot**")
        colm1, colm2 = st.columns(2)
        with colm1:
            dow_name = st.selectbox("Day", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
            start_t = st.time_input("Start", value=dtime(9,0))
        with colm2:
            end_t = st.time_input("End", value=dtime(9,30))
            pinned = st.selectbox("Pin a specific topic? (optional)",
                                  ["(none)"] + [f"{tid}: {t}" for tid,t,_ in topics_for_select])
        if st.button("Add Slot"):
            dow_idx = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(dow_name)
            pin_tid = None if pinned == "(none)" else int(pinned.split(":")[0])
            if start_t >= end_t:
                st.warning("End must be after start.")
            else:
                add_manual_slot(uid, subject_id, dow_idx, start_t.strftime("%H:%M"), end_t.strftime("%H:%M"), pin_tid)
                st.success("Slot added.")
                st.rerun()

# --- Load subject-specific data ---
topics = get_topics(subject_id)
if not topics:
    st.info("No topics in this subject yet. Use Fetch Syllabus or add topics.")
    st.stop()

events = get_quiz_events(uid, subject_id)
last_seen = {}
for tid, ts, correct, _ in events:
    try:
        last_seen[tid] = datetime.fromisoformat(ts)
    except Exception:
        pass

topic_idx = {tid:i for i,(tid,_,_) in enumerate(topics)}
mastery = train_and_estimate_mastery(topics, events, topic_idx)
bstats = bandit_stats(uid, subject_id)

# --- Generate this week's plan if absent ---
mon = date.today() - timedelta(days=date.today().weekday())
sun = mon + timedelta(days=6)
generate_week_plan(uid, subject_id, topics, mastery, last_seen, bstats, planner_mode)

# --- Fetch sessions for week ---
wk_sessions = sessions_for_week(uid, subject_id, mon, sun)
sess_df = pd.DataFrame(wk_sessions, columns=["id","date","time","topic_id","duration","status"])
topic_map = {tid:t for tid,t,_ in topics}
diff_map  = {tid:df for tid,_,df in topics}
if not sess_df.empty:
    sess_df["Topic"] = sess_df["topic_id"].map(topic_map)
    sess_df["Difficulty"] = sess_df["topic_id"].map(diff_map)

# ====== Today’s Plan ======
colA, colB = st.columns([1,1])
with colA:
    st.subheader(f"Today · {date.today().isoformat()} · {chosen_name}")
    if not sess_df.empty:
        today_df = sess_df[sess_df["date"] == date.today().isoformat()].copy()
        if today_df.empty:
            st.write("No sessions today.")
        else:
            for _, row in today_df.sort_values("time").iterrows():
                st.markdown(f"**{row['time']}** · {row['Topic']} (d{row['Difficulty']}) — {row['duration']} min")
                c1, c2, c3 = st.columns([0.2,0.4,0.4])
                with c1:
                    if st.button("Done", key=f"done_{row['id']}"):
                        update_session_status(int(row["id"]), "done")
                        insert_quiz(uid, int(row["topic_id"]), 1, row["duration"])
                        insert_bandit(uid, int(row["topic_id"]), reward(1, row["duration"]))
                        st.rerun()
                with c2:
                    if st.button("Skip", key=f"skip_{row['id']}"):
                        update_session_status(int(row["id"]), "skipped")
                        st.rerun()
                with c3:
                    st.write("Status:", row["status"])
    else:
        st.write("Planner is empty. Generate by fetching syllabus or switching modes.")

# ====== Week View ======
with colB:
    st.subheader(f"Week · {mon.isoformat()} → {sun.isoformat()} · {chosen_name}")
    if sess_df.empty:
        st.write("No sessions scheduled.")
    else:
        display = []
        for _, row in sess_df.sort_values(["date","time"]).iterrows():
            display.append({
                "Date": row["date"],
                "Time": row["time"],
                "Topic": row["Topic"],
                "Dur (min)": row["duration"],
                "Status": row["status"]
            })
        st.dataframe(pd.DataFrame(display), use_container_width=True, hide_index=True)
        st.caption("Legend: pending / done / skipped")

st.divider()

# ====== Progress Charts ======
st.subheader(f"Progress · {chosen_name}")
if not sess_df.empty:
    summary = sess_df.groupby("status")["id"].count().reindex(["pending","done","skipped"]).fillna(0)
    fig1, ax1 = plt.subplots()
    ax1.bar(summary.index, summary.values)
    ax1.set_title("Weekly Sessions by Status")
    ax1.set_xlabel("Status"); ax1.set_ylabel("Count")
    st.pyplot(fig1)

if events:
    ev_df = pd.DataFrame(events, columns=["topic_id","ts","correct","time_sec"])
    ev_df["ts"] = pd.to_datetime(ev_df["ts"])
    curves=[]
    for tid in ev_df["topic_id"].unique():
        sub = ev_df[ev_df["topic_id"]==tid].sort_values("ts")
        roll = sub["correct"].rolling(window=3, min_periods=1).mean()
        curves.append(pd.DataFrame({"ts": sub["ts"], "topic_id": tid, "rolling_acc": roll}))
    tr = pd.concat(curves).sort_values("ts")
    fig2, ax2 = plt.subplots()
    for tid in tr["topic_id"].unique():
        s = tr[tr["topic_id"]==tid]
        ax2.plot(s["ts"], s["rolling_acc"], label=topic_map.get(tid,f"Topic {tid}"))
    ax2.set_ylim(0,1); ax2.set_title("Rolling Accuracy (0–1)")
    ax2.set_xlabel("Time"); ax2.set_ylabel("Accuracy")
    ax2.legend(loc="best")
    st.pyplot(fig2)
else:
    st.caption("No attempts yet → charts appear after you mark sessions Done.")

st.divider()

# ====== Manual Attempt ======
st.subheader("Log a Custom Attempt")
pick = st.selectbox("Topic", [f"{tid}: {t}" for tid,t,_ in topics])
chosen_tid = int(pick.split(":")[0])
correct = st.checkbox("Correct?", value=True)
tsec = st.number_input("Time (sec)", value=120.0, min_value=1.0, step=1.0)
if st.button("Add Attempt"):
    insert_quiz(uid, chosen_tid, int(correct), float(tsec))
    insert_bandit(uid, chosen_tid, reward(int(correct), float(tsec)))
    st.success("Recorded.")
    st.rerun()

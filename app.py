import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from collections import deque

# ===============================================
# Config
# ===============================================
FREE_SECONDS_PER_DAY = 15 * 60
INTERACTION_COST = 8  # burn per graded check
NEXT_CLICK_COST = 2

# ===============================================
# Seed questions (extend freely)
# Fields:
# id, tag, prompt, choices, answer, calc ("No"/"Yes"), desmos (True/False), exp_intuitive, exp_formal
# ===============================================
QUESTIONS = [
    # -------- Algebra (No-Calc friendly) --------
    dict(id=1, tag="algebra", calc="No", desmos=False,
         prompt="Solve for x: 2x + 3 = 11",
         choices=["3", "4", "5", "8"], answer="4",
         exp_intuitive="Balance both sides: remove 3, then split by 2.",
         exp_formal="2x+3=11 ⇒ 2x=8 ⇒ x=4."),
    dict(id=2, tag="algebra", calc="No", desmos=False,
         prompt="If (x−2)(x+3)=0, solutions are:",
         choices=["x=2 only", "x=−3 only", "x=2 or x=−3", "No solution"], answer="x=2 or x=−3",
         exp_intuitive="Zero product → one factor must be zero.",
         exp_formal="x−2=0 or x+3=0 ⇒ x=2, −3."),
    dict(id=3, tag="algebra", calc="No", desmos=True,
         prompt="Slope of y = 5x − 7 is:",
         choices=["−7", "0", "5", "12"], answer="5",
         exp_intuitive="y=mx+b; m is the tilt.",
         exp_formal="Slope-intercept form ⇒ m=5."),
    dict(id=4, tag="algebra", calc="No", desmos=False,
         prompt="Solve: 3(2x−1)=15",
         choices=["x=2", "x=3", "x=8/3", "x=5/3"], answer="x=3",
         exp_intuitive="Distribute then isolate x.",
         exp_formal="6x−3=15 ⇒ 6x=18 ⇒ x=3."),

    # -------- Functions --------
    dict(id=5, tag="functions", calc="No", desmos=True,
         prompt="Average rate of change of f(x)=x^2 on [2,5]:",
         choices=["3", "7", "9", "11"], answer="7",
         exp_intuitive="Rise/run: (25−4)/(5−2)=21/3=7.",
         exp_formal="(f(5)−f(2))/(5−2)=(25−4)/3=7."),
    dict(id=6, tag="functions", calc="No", desmos=True,
         prompt="If f(x)=2x+1 and g(x)=x^2, then (f∘g)(3) = ?",
         choices=["19", "20", "21", "22"], answer="19",
         exp_intuitive="Do g first: g(3)=9; then f(9)=19.",
         exp_formal="(f∘g)(x)=2x^2+1 ⇒ at x=3 gives 19."),
    dict(id=7, tag="functions", calc="No", desmos=True,
         prompt="Vertex of y=(x−4)^2+5 is:",
         choices=["(4,5)", "(−4,5)", "(4,−5)", "(−4,−5)"], answer="(4,5)",
         exp_intuitive="Shift right 4, up 5.",
         exp_formal="Vertex form y=(x−h)^2+k ⇒ (4,5)."),
    dict(id=8, tag="functions", calc="Yes", desmos=True,
         prompt="For f(x)=ax+b, if f(2)=7 and f(5)=16, then a=?",
         choices=["2", "3", "4", "9/2"], answer="3",
         exp_intuitive="Slope = (16−7)/(5−2)=9/3=3.",
         exp_formal="a=(f(5)−f(2))/(5−2)=3."),

    # -------- Data Analysis --------
    dict(id=9, tag="data", calc="Yes", desmos=False,
         prompt="A dataset has mean 10. Adding a value 12 makes the mean:",
         choices=["Slightly above 10", "Exactly 10", "Slightly below 10", "Cannot tell"], answer="Slightly above 10",
         exp_intuitive="Adding > mean pulls mean upward.",
         exp_formal="(S+12)/(n+1) > 10 when 12>10."),
    dict(id=10, tag="data", calc="No", desmos=False,
         prompt="Median is resistant to outliers. True/False?",
         choices=["True", "False"], answer="True",
         exp_intuitive="One extreme hardly moves the middle.",
         exp_formal="Median depends on order, not magnitudes."),
    dict(id=11, tag="data", calc="Yes", desmos=False,
         prompt="In a bar chart, which measure do outliers affect most?",
         choices=["Mean", "Median", "Mode", "IQR"], answer="Mean",
         exp_intuitive="Mean ‘feels’ every value.",
         exp_formal="Mean is non-resistant; median/IQR more robust."),
    dict(id=12, tag="data", calc="Yes", desmos=False,
         prompt="Standard deviation measures:",
         choices=["Center", "Spread", "Skew", "Sample size"], answer="Spread",
         exp_intuitive="How far typical values are from mean.",
         exp_formal="SD quantifies dispersion."),

    # -------- Geometry --------
    dict(id=13, tag="geometry", calc="No", desmos=True,
         prompt="Area of right triangle with legs 6 and 8:",
         choices=["14", "24", "28", "48"], answer="24",
         exp_intuitive="Half of 6×8 rectangle.",
         exp_formal="A=(1/2)ab=24."),
    dict(id=14, tag="geometry", calc="No", desmos=True,
         prompt="Circumference of a circle with radius 3:",
         choices=["6π", "9π", "3π", "12π"], answer="6π",
         exp_intuitive="Wrap radius: 2πr.",
         exp_formal="C=2πr=6π."),
    dict(id=15, tag="geometry", calc="No", desmos=False,
         prompt="A line has slope 2 and passes through (0,−1). Equation?",
         choices=["y=2x−1", "y=2x+1", "y=−2x−1", "y=x−1"], answer="y=2x−1",
         exp_intuitive="Start at −1, tilt 2 per 1 right.",
         exp_formal="y=mx+b ⇒ y=2x−1."),
    dict(id=16, tag="geometry", calc="Yes", desmos=True,
         prompt="Distance between (−2,1) and (4,4):",
         choices=["√13", "√45", "√40", "7"], answer="√45",
         exp_intuitive="dx=6, dy=3 → √(36+9)=√45.",
         exp_formal="Distance formula √((x2−x1)^2+(y2−y1)^2)."),

    # -------- Trigonometry --------
    dict(id=17, tag="trig", calc="No", desmos=False,
         prompt="sin(30°) = ?",
         choices=["1/2", "√3/2", "√2/2", "0"], answer="1/2",
         exp_intuitive="30-60-90 triangle: short/hyp.",
         exp_formal="sin 30°=1/2."),
    dict(id=18, tag="trig", calc="No", desmos=False,
         prompt="tan(45°) = ?",
         choices=["0", "1", "√3", "1/√3"], answer="1",
         exp_intuitive="Isosceles right triangle: opp=adj.",
         exp_formal="tan 45°=1."),
    dict(id=19, tag="trig", calc="No", desmos=True,
         prompt="cos(60°) = ?",
         choices=["1/2", "√3/2", "0", "√2/2"], answer="1/2",
         exp_intuitive="30-60-90: adj/hyp at 60° is 1/2.",
         exp_formal="cos 60°=1/2."),
    dict(id=20, tag="trig", calc="No", desmos=False,
         prompt="If sin θ=3/5 with θ acute, cos θ = ?",
         choices=["4/5", "3/4", "5/3", "√2/2"], answer="4/5",
         exp_intuitive="3-4-5 triangle.",
         exp_formal="cos θ=√(1−sin²θ)=√(1−9/25)=4/5."),
]

DF = pd.DataFrame(QUESTIONS)

# ===============================================
# Helpers: time limit
# ===============================================
def time_left_today():
    today = str(date.today())
    if "quota" not in st.session_state or today not in st.session_state.quota:
        st.session_state.quota = {today: FREE_SECONDS_PER_DAY}
    return st.session_state.quota[today]

def spend_time(sec: int):
    today = str(date.today())
    st.session_state.quota[today] = max(0, st.session_state.quota[today] - int(sec))

# ===============================================
# Adaptive selection & review queue
# ===============================================
def init_state():
    if "history" not in st.session_state:
        st.session_state.history = pd.DataFrame({
            "id": pd.Series(dtype="int64"),
            "tag": pd.Series(dtype="string"),
            "correct": pd.Series(dtype="float64"),
            "calc": pd.Series(dtype="string"),
            "desmos": pd.Series(dtype="boolean"),
        })
    if "phase" not in st.session_state:
        st.session_state.phase = "ask"  # ask | feedback
    if "current" not in st.session_state:
        st.session_state.current = None
    if "feedback" not in st.session_state:
        st.session_state.feedback = None
    if "review" not in st.session_state:
        st.session_state.review = deque()  # queue of ids to retry later (spaced)
    if "skip_desmos" not in st.session_state:
        st.session_state.skip_desmos = False
    if "mode_calc" not in st.session_state:
        st.session_state.mode_calc = "Both"
    if "style" not in st.session_state:
        st.session_state.style = "Intuitive"

def accuracy_by_tag():
    hist = st.session_state.history
    if hist.empty:
        return {}
    means = (hist.groupby("tag")["correct"].mean()).to_dict()
    return means

def weakest_tag():
    acc = accuracy_by_tag()
    tags = list(DF["tag"].unique())
    if not acc:
        return np.random.choice(tags)
    # tags attempted get their accuracy; unattempted treated as 1.0 (so we don't overly force them)
    scored = []
    for t in tags:
        a = acc.get(t, 1.0)
        n = int((st.session_state.history["tag"] == t).sum())
        scored.append((a, n, t))
    scored.sort()  # lowest accuracy, then fewer attempts
    return scored[0][2]

def candidate_pool():
    pool = DF.copy()
    if st.session_state.skip_desmos:
        pool = pool[pool["desmos"] == False]
    mc = st.session_state.mode_calc
    if mc in ("Yes", "No"):
        pool = pool[pool["calc"] == mc]
    return pool

def choose_next():
    # 1) Prioritize review queue (if any)
    if st.session_state.review:
        rid = st.session_state.review.popleft()
        row = DF[DF["id"] == rid]
        if not row.empty:
            return row.sample(1).iloc[0].to_dict()

    pool = candidate_pool()
    seen = set(st.session_state.history["id"].tolist())
    pool = pool[~pool["id"].isin(seen)]
    weak = weakest_tag()
    weak_pool = pool[pool["tag"] == weak]
    if not weak_pool.empty:
        return weak_pool.sample(1).iloc[0].to_dict()
    if not pool.empty:
        return pool.sample(1).iloc[0].to_dict()
    # recycle if all seen (MVP)
    pool = candidate_pool()
    return pool.sample(1).iloc[0].to_dict()

def record_result(q, correct: bool):
    row = pd.DataFrame([{
        "id": int(q["id"]),
        "tag": str(q["tag"]),
        "correct": float(1 if correct else 0),
        "calc": str(q["calc"]),
        "desmos": bool(q["desmos"]),
    }])
    st.session_state.history = pd.concat([st.session_state.history, row], ignore_index=True)
    if not correct:
        # push back for review after a few items
        st.session_state.review.append(q["id"])

# ===============================================
# UI
# ===============================================
st.set_page_config(page_title="SAT Math MVP", page_icon="🧠", layout="wide")
init_state()

colL, colR = st.columns([2,1])
with colL:
    st.title("🧠 SAT Math — Adaptive MVP")
    st.caption("Learn your way (Intuitive ↔ Formal). 15-min/day focus timer. Review your mistakes.")

with colR:
    remaining = time_left_today()
    m, s = remaining // 60, remaining % 60
    st.metric("Free time left today", f"{m}m {s}s")

with st.sidebar:
    st.header("Session")
    st.session_state.style = st.radio("Explanation style", ["Intuitive", "Formal"], horizontal=False, index=0)
    st.session_state.mode_calc = st.radio("Calculator section", ["Both", "No", "Yes"], index=0,
                                          help="Filter by SAT 'No-Calculator' vs 'Calculator' sections.")
    st.session_state.skip_desmos = st.checkbox("Hide Desmos-solvable (skip easy graphable)", value=False)
    if st.button("Reset progress"):
        st.session_state.history = st.session_state.history.iloc[0:0]
        st.session_state.review.clear()
        st.session_state.phase = "ask"
        st.session_state.current = None
        st.session_state.feedback = None
        st.rerun()
    if st.button("Export mistakes CSV"):
        errs = st.session_state.history[st.session_state.history["correct"] == 0.0]
        errs = errs.merge(DF[["id","prompt","tag","calc","desmos"]], on="id", how="left")
        csv = errs.to_csv(index=False)
        st.download_button("Download mistakes.csv", data=csv, file_name="sat_mistakes.csv", mime="text/csv", use_container_width=True)

# stop if out of time
if remaining <= 0:
    st.error("⏳ You used your free 15 minutes today. Come back tomorrow.")
    st.stop()

# Pick current question if needed
if st.session_state.current is None:
    st.session_state.current = choose_next()
q = st.session_state.current

# ASK PHASE
if st.session_state.phase == "ask":
    with st.form(key=f"form_{q['id']}"):
        st.subheader("Question")
        top = st.columns([6,2,2,2])
        with top[0]:
            st.write(q["prompt"])
        with top[1]:
            st.caption(f"Tag: **{q['tag']}**")
        with top[2]:
            st.caption(f"Calc: **{q['calc']}**")
        with top[3]:
            st.caption(f"Desmos: **{'Yes' if q['desmos'] else 'No'}**")

        choice = st.radio("Pick one:", q["choices"], index=None, key=f"choice_{q['id']}")
        submitted = st.form_submit_button("Check answer")
        if submitted:
            if choice is None:
                st.warning("Pick an option first.")
            else:
                correct = (choice == q["answer"])
                record_result(q, correct)
                st.session_state.feedback = {"correct": correct, "chosen": choice}
                st.session_state.phase = "feedback"
                spend_time(INTERACTION_COST)
                st.rerun()

# FEEDBACK PHASE
if st.session_state.phase == "feedback":
    fb = st.session_state.feedback or {"correct": False, "chosen": None}
    if fb["correct"]:
        st.success("✅ Correct!")
    else:
        st.error(f"❌ Not quite. Correct answer: **{q['answer']}**")

    # Explanation
    exp = q["exp_intuitive"] if st.session_state.style == "Intuitive" else q["exp_formal"]
    st.markdown("### Explanation — " + st.session_state.style)
    st.write(exp)

    st.divider()
    cols = st.columns([1,1,4])
    with cols[0]:
        if st.button("Next question →", use_container_width=True):
            st.session_state.current = choose_next()
            st.session_state.phase = "ask"
            st.session_state.feedback = None
            st.session_state.pop(f"choice_{q['id']}", None)
            spend_time(NEXT_CLICK_COST)
            st.rerun()
    with cols[1]:
        if st.button("Add to review again", use_container_width=True):
            st.session_state.review.append(q["id"])
            st.toast("Queued for review")
    with cols[2]:
        pass

st.divider()

# Progress panel
st.markdown("### Your weak areas")
hist = st.session_state.history
if hist.empty:
    st.info("Answer some questions to see your stats.")
else:
    tag_means = (hist.groupby("tag")["correct"].mean().sort_values())
    stats_df = tag_means.round(2).rename("accuracy").reset_index()
    weak = stats_df.iloc[0]["tag"]
    st.write(f"**{weak}** is currently your lowest-accuracy tag.")
    st.dataframe(stats_df, use_container_width=True)

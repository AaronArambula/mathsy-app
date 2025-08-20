import os, ast, math, json
from collections import deque
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

# ===============================================
# Config
# ===============================================
FREE_SECONDS_PER_DAY = 15 * 60
INTERACTION_COST = 8     # seconds burned per graded check
NEXT_CLICK_COST = 2      # seconds burned when advancing
BANK_PATH = "sat_bank.csv"   # optional external bank (CSV or JSON)

# ===============================================
# Built-in seed bank (used if CSV/JSON missing)
# ===============================================
SEED = [
    # id, tag, calc, desmos, difficulty, prompt, choices, answer, exp_intuitive, exp_formal, exp_visual_key, exp_kinesthetic_key
    dict(id=1, tag="algebra", calc="No", desmos=False, difficulty=1,
         prompt="Solve for x: 2x + 3 = 11",
         choices=["3","4","5","8"], answer="4",
         exp_intuitive="Balance both sides: subtract 3, then divide by 2.",
         exp_formal="2x+3=11 ⇒ 2x=8 ⇒ x=4.",
         exp_visual_key="algebra_balance", exp_kinesthetic_key="algebra_step_moves"),
    dict(id=2, tag="algebra", calc="No", desmos=False, difficulty=1,
         prompt="If (x−2)(x+3)=0, solutions are:",
         choices=["x=2 only","x=−3 only","x=2 or x=−3","No solution"], answer="x=2 or x=−3",
         exp_intuitive="Zero product → one factor must be zero.",
         exp_formal="x−2=0 or x+3=0 ⇒ x=2, −3.",
         exp_visual_key="", exp_kinesthetic_key=""),
    dict(id=3, tag="algebra", calc="No", desmos=True, difficulty=1,
         prompt="Slope of y = 5x − 7 is:",
         choices=["−7","0","5","12"], answer="5",
         exp_intuitive="In y=mx+b, m is the tilt.",
         exp_formal="Slope-intercept form ⇒ m=5.",
         exp_visual_key="", exp_kinesthetic_key=""),
    dict(id=4, tag="functions", calc="No", desmos=True, difficulty=2,
         prompt="Average rate of change of f(x)=x^2 on [2,5]:",
         choices=["3","7","9","11"], answer="7",
         exp_intuitive="Rise/run: (25−4)/(5−2)=21/3=7.",
         exp_formal="(f(5)−f(2))/(5−2)=(25−4)/3=7.",
         exp_visual_key="roc_parabola", exp_kinesthetic_key=""),
    dict(id=5, tag="functions", calc="No", desmos=True, difficulty=2,
         prompt="If f(x)=2x+1 and g(x)=x^2, then (f∘g)(3) = ?",
         choices=["19","20","21","22"], answer="19",
         exp_intuitive="Do g first: g(3)=9; then f(9)=19.",
         exp_formal="(f∘g)(x)=2x^2+1 ⇒ at x=3 gives 19.",
         exp_visual_key="", exp_kinesthetic_key=""),
    dict(id=6, tag="functions", calc="No", desmos=True, difficulty=1,
         prompt="Vertex of y=(x−4)^2+5 is:",
         choices=["(4,5)","(−4,5)","(4,−5)","(−4,−5)"], answer="(4,5)",
         exp_intuitive="Shift right 4, up 5.",
         exp_formal="Vertex form y=(x−h)^2+k ⇒ (4,5).",
         exp_visual_key="", exp_kinesthetic_key=""),
    dict(id=7, tag="data", calc="Yes", desmos=False, difficulty=1,
         prompt="A dataset has mean 10. Adding a value 12 makes the mean:",
         choices=["Slightly above 10","Exactly 10","Slightly below 10","Cannot tell"], answer="Slightly above 10",
         exp_intuitive="Adding > mean pulls mean upward.",
         exp_formal="(S+12)/(n+1) > 10 when 12>10.",
         exp_visual_key="", exp_kinesthetic_key=""),
    dict(id=8, tag="data", calc="No", desmos=False, difficulty=1,
         prompt="Median is resistant to outliers. True/False?",
         choices=["True","False"], answer="True",
         exp_intuitive="One extreme hardly moves the middle.",
         exp_formal="Median depends on order, not magnitudes.",
         exp_visual_key="", exp_kinesthetic_key=""),
    dict(id=9, tag="geometry", calc="No", desmos=True, difficulty=1,
         prompt="Area of right triangle with legs 6 and 8:",
         choices=["14","24","28","48"], answer="24",
         exp_intuitive="Half of 6×8 rectangle.",
         exp_formal="A=(1/2)ab=24.",
         exp_visual_key="", exp_kinesthetic_key=""),
    dict(id=10, tag="geometry", calc="No", desmos=True, difficulty=2,
         prompt="Distance between (−2,1) and (4,4):",
         choices=["√13","√45","√40","7"], answer="√45",
         exp_intuitive="dx=6, dy=3 → √(36+9)=√45.",
         exp_formal="Distance = √((x2−x1)^2+(y2−y1)^2).",
         exp_visual_key="distance_plot", exp_kinesthetic_key=""),
    dict(id=11, tag="trig", calc="No", desmos=False, difficulty=1,
         prompt="sin(30°) = ?",
         choices=["1/2","√3/2","√2/2","0"], answer="1/2",
         exp_intuitive="30-60-90: short/hyp.",
         exp_formal="sin 30°=1/2.",
         exp_visual_key="", exp_kinesthetic_key="unit_circle"),
    dict(id=12, tag="trig", calc="No", desmos=False, difficulty=1,
         prompt="tan(45°) = ?",
         choices=["0","1","√3","1/√3"], answer="1",
         exp_intuitive="Isosceles right triangle: opp=adj.",
         exp_formal="tan 45°=1.",
         exp_visual_key="", exp_kinesthetic_key="unit_circle"),
]

SEED_DF = pd.DataFrame(SEED)

# ===============================================
# Load external bank (CSV/JSON) or fallback
# ===============================================
def load_questions_dataframe():
    if os.path.exists(BANK_PATH):
        ext = os.path.splitext(BANK_PATH)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(BANK_PATH)
            # Required columns
            req = ["id","tag","calc","desmos","difficulty","prompt","choices","answer","exp_intuitive","exp_formal"]
            missing = [c for c in req if c not in df.columns]
            if missing:
                st.warning(f"{BANK_PATH} missing columns: {missing}. Falling back to seed.")
                return SEED_DF.copy()

            # Parse choices (stored as JSON list in CSV)
            df["choices"] = df["choices"].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)
            # Coerce types
            df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
            df["difficulty"] = pd.to_numeric(df["difficulty"], errors="coerce").fillna(2).astype(int)
            df["desmos"] = df["desmos"].astype(str).str.lower().isin(["true","1","yes"])
            # Optional columns
            for c in ["exp_visual_key","exp_kinesthetic_key"]:
                if c not in df.columns: df[c] = ""
            return df
        elif ext == ".json":
            df = pd.read_json(BANK_PATH)
            if "choices" in df.columns:
                df["choices"] = df["choices"].apply(lambda x: x if isinstance(x, list) else [])
            if "difficulty" not in df.columns: df["difficulty"] = 2
            if "desmos" in df.columns:
                df["desmos"] = df["desmos"].astype(bool)
            else:
                df["desmos"] = False
            for c in ["exp_visual_key","exp_kinesthetic_key"]:
                if c not in df.columns: df[c] = ""
            return df
        else:
            st.warning(f"Unsupported bank format: {ext}. Using seed questions.")
            return SEED_DF.copy()
    return SEED_DF.copy()

DF = load_questions_dataframe()

# ===============================================
# Time quota helpers
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
# State init
# ===============================================
def init_state():
    if "history" not in st.session_state:
        st.session_state.history = pd.DataFrame({
            "id": pd.Series(dtype="int64"),
            "tag": pd.Series(dtype="string"),
            "correct": pd.Series(dtype="float64"),
            "calc": pd.Series(dtype="string"),
            "desmos": pd.Series(dtype="boolean"),
            "difficulty": pd.Series(dtype="int64"),
        })
    if "phase" not in st.session_state:
        st.session_state.phase = "ask"  # ask | feedback
    if "current" not in st.session_state:
        st.session_state.current = None
    if "feedback" not in st.session_state:
        st.session_state.feedback = None
    if "review" not in st.session_state:
        st.session_state.review = deque()
    if "skip_desmos" not in st.session_state:
        st.session_state.skip_desmos = False
    if "mode_calc" not in st.session_state:
        st.session_state.mode_calc = "Both"
    if "style" not in st.session_state:
        st.session_state.style = "Intuitive"

def record_result(q, correct: bool):
    row = pd.DataFrame([{
        "id": int(q["id"]),
        "tag": str(q["tag"]),
        "correct": float(1 if correct else 0),
        "calc": str(q.get("calc","Both")),
        "desmos": bool(q.get("desmos", False)),
        "difficulty": int(q.get("difficulty", 2)),
    }])
    st.session_state.history = pd.concat([st.session_state.history, row], ignore_index=True)
    if not correct:
        # schedule a revisit
        st.session_state.review.append(q["id"])

def accuracy_by_tag():
    hist = st.session_state.history
    if hist.empty:
        return {}
    return (hist.groupby("tag")["correct"].mean()).to_dict()

def weakest_tag():
    acc = accuracy_by_tag()
    tags = list(DF["tag"].unique())
    if not tags:
        return None
    if not acc:
        return np.random.choice(tags)
    # prioritize lowest accuracy; tie-breaker by fewer attempts
    scored = []
    for t in tags:
        a = acc.get(t, 1.0)  # unattempted appear strong to avoid tunnel vision
        n = int((st.session_state.history["tag"] == t).sum())
        scored.append((a, n, t))
    scored.sort()  # lowest accuracy, then fewer attempts
    return scored[0][2]

def candidate_pool():
    pool = DF.copy()
    if st.session_state.skip_desmos and "desmos" in pool.columns:
        pool = pool[pool["desmos"] == False]
    mc = st.session_state.mode_calc
    if "calc" in pool.columns and mc in ("Yes","No"):
        pool = pool[pool["calc"] == mc]
    return pool

def choose_next():
    # 1) review has priority
    if st.session_state.review:
        rid = st.session_state.review.popleft()
        row = DF[DF["id"] == rid]
        if not row.empty:
            return row.sample(1).iloc[0].to_dict()

    pool = candidate_pool()
    if pool.empty:
        return None

    # 2) target weakest tag
    weak = weakest_tag()
    if weak in pool["tag"].unique():
        pool = pool[pool["tag"] == weak]

    # 3) difficulty-aware nudge
    tag_acc = accuracy_by_tag().get(weak, 0.0)
    if "difficulty" in pool.columns and not pool.empty:
        if tag_acc < 0.6:
            filt = pool["difficulty"].between(1, 3)
        else:
            filt = pool["difficulty"].between(3, 5)
        if filt.any():
            pool = pool[filt]

    # 4) avoid immediate repeats
    seen = set(st.session_state.history["id"].tolist())
    cand = pool[~pool["id"].isin(seen)]
    if not cand.empty:
        return cand.sample(1).iloc[0].to_dict()
    # recycle if exhausted
    return pool.sample(1).iloc[0].to_dict()

# ===============================================
# Visual explainers (registry by key)
# ===============================================
def viz_algebra_balance(q):
    st.markdown("**Balance scale intuition**")
    st.caption("Try values of x and see when both sides match.")
    x = st.slider("Try x", -5.0, 10.0, 2.0, 0.5, key=f"bal_{q['id']}")
    lhs = 2*x + 3
    rhs = 11
    st.write(f"LHS = 2·{x:.1f} + 3 = **{lhs:.1f}**, RHS = **{rhs}**")
    st.progress(float(max(0.0, 1.0 - min(1.0, abs(lhs-rhs)/10.0))))
    if abs(lhs-rhs) < 1e-6:
        st.success("Balanced — that x solves it.")

def viz_roc_parabola(q):
    st.markdown("**Average rate of change on f(x)=x²**")
    a = st.slider("Interval length (x₂−x₁)", 1, 6, 3, key=f"roc_a_{q['id']}")
    x1 = st.slider("Left point x₁", -4, 4, 2, key=f"roc_x1_{q['id']}")
    x2 = x1 + a
    y1, y2 = x1**2, x2**2
    slope = (y2 - y1) / (x2 - x1)
    st.write(f"A({x1},{y1}) → B({x2},{y2}); secant slope = **{slope:.2f}**")
    xs = np.linspace(-6, 6, 241)
    ys = xs**2
    df = pd.DataFrame({"x": xs, "f(x)=x^2": ys}).set_index("x")
    st.line_chart(df)
    st.caption("Secant slope = average rate over [x₁, x₂]. As x₂→x₁, it approaches the derivative.")

def viz_distance_plot(q):
    st.markdown("**Distance between two points**")
    x1 = st.number_input("x₁", value=-2.0, step=1.0, key=f"dpx1_{q['id']}")
    y1 = st.number_input("y₁", value=1.0, step=1.0, key=f"dpy1_{q['id']}")
    x2 = st.number_input("x₂", value=4.0, step=1.0, key=f"dpx2_{q['id']}")
    y2 = st.number_input("y₂", value=4.0, step=1.0, key=f"dpy2_{q['id']}")
    dist = math.hypot(x2-x1, y2-y1)
    st.write(f"Distance = **√(({x2}−{x1})²+({y2}−{y1})²) = {dist:.3f}**")

VISUALS = {
    "algebra_balance": viz_algebra_balance,
    "roc_parabola": viz_roc_parabola,
    "distance_plot": viz_distance_plot,
}

# ===============================================
# Kinesthetic explainers (registry by key)
# ===============================================
def kine_algebra_step_moves(q):
    st.markdown("**Solve 2x+3=11 by legal moves**")
    step = st.radio("Pick the next valid move:", [
        "Add 3 to both sides",
        "Subtract 3 from both sides",
        "Multiply both sides by 3",
        "Divide both sides by 2",
    ], index=None, key=f"kstep_{q['id']}")
    if st.button("Check move", key=f"kbtn_{q['id']}"):
        if step == "Subtract 3 from both sides":
            st.success("Correct: 2x+3−3=11−3 → 2x=8.")
            st.info("Next: divide both sides by 2.")
        else:
            st.error("Not the best next step to isolate x.")

def kine_unit_circle(q):
    st.markdown("**Unit circle — feel sin/cos**")
    angle = st.slider("θ (degrees)", 0, 360, 30, key=f"kuc_{q['id']}")
    rad = math.radians(angle)
    st.write(f"cos θ ≈ {math.cos(rad):.3f},  sin θ ≈ {math.sin(rad):.3f}")
    st.caption("Drag θ; x-projection = cos, y-projection = sin.")

KINE = {
    "algebra_step_moves": kine_algebra_step_moves,
    "unit_circle": kine_unit_circle,
}

# ===============================================
# Streamlit UI
# ===============================================
st.set_page_config(page_title="SAT Math MVP", page_icon="🧠", layout="wide")
init_state()

colL, colR = st.columns([2,1])
with colL:
    st.title("🧠 SAT Math — Adaptive MVP")
    st.caption("Adaptive next-question • Intuitive ↔ Formal • Visual & Kinesthetic explainers • 15-min/day focus")
with colR:
    remaining = time_left_today()
    st.metric("Free time left today", f"{remaining//60}m {remaining%60}s")

with st.sidebar:
    st.header("Session")
    st.session_state.style = st.radio("Explanation style", ["Intuitive","Formal"], index=0)
    st.session_state.mode_calc = st.radio("Calculator section", ["Both","No","Yes"], index=0)
    st.session_state.skip_desmos = st.checkbox("Hide Desmos-solvable", value=False)
    if st.button("Reset progress"):
        st.session_state.history = st.session_state.history.iloc[0:0]
        st.session_state.review.clear()
        st.session_state.phase = "ask"
        st.session_state.current = None
        st.session_state.feedback = None
        st.rerun()
    if st.button("Export mistakes CSV"):
        errs = st.session_state.history[st.session_state.history["correct"] == 0.0]
        errs = errs.merge(DF[["id","prompt","tag","calc","desmos","difficulty"]], on="id", how="left")
        csv = errs.to_csv(index=False)
        st.download_button("Download mistakes.csv", data=csv, file_name="sat_mistakes.csv", mime="text/csv", use_container_width=True)

# stop if out of time
if remaining <= 0:
    st.error("⏳ You used your free 15 minutes today. Come back tomorrow.")
    st.stop()

# current question
if st.session_state.current is None:
    st.session_state.current = choose_next()
q = st.session_state.current
if q is None:
    st.error("No questions available. Check your bank file.")
    st.stop()

# ASK PHASE
if st.session_state.phase == "ask":
    with st.form(key=f"form_{q['id']}"):
        st.subheader("Question")
        top = st.columns([6,2,2,2])
        with top[0]:
            st.write(q["prompt"])
        with top[1]:
            st.caption(f"Tag: **{q.get('tag','?')}**")
        with top[2]:
            st.caption(f"Calc: **{q.get('calc','Both')}**")
        with top[3]:
            st.caption(f"Desmos: **{'Yes' if q.get('desmos', False) else 'No'}**")

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
    exp = q.get("exp_intuitive") if st.session_state.style == "Intuitive" else q.get("exp_formal")
    st.markdown("### Explanation — " + st.session_state.style)
    st.write(exp or "No explanation provided.")

    # Visual explainer
    vk = q.get("exp_visual_key", "")
    if vk and vk in VISUALS:
        with st.expander("See it visually"):
            VISUALS[vk](q)

    # Kinesthetic explainer
    kk = q.get("exp_kinesthetic_key", "")
    if kk and kk in KINE:
        with st.expander("Try it yourself (kinesthetic)"):
            KINE[kk](q)

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

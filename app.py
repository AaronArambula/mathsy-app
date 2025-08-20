import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

# ---------------------------
# Minimal content
# ---------------------------
QUESTIONS = pd.DataFrame([
    {"id": 1, "tag": "algebra_linear", "prompt": "Solve for x: 2x + 3 = 11", "choices": ["3", "4", "5", "8"], "answer": "4"},
    {"id": 2, "tag": "algebra_linear", "prompt": "Slope of y = 5x - 7 is:", "choices": ["-7", "0", "5", "12"], "answer": "5"},
    {"id": 3, "tag": "rates", "prompt": "Average rate of change of f(x)=x^2 from x=2 to x=5:", "choices": ["3", "7", "9", "11"], "answer": "7"},
    {"id": 4, "tag": "derivative", "prompt": "Derivative of f(x)=x^2 is:", "choices": ["x", "2x", "x^3", "2"], "answer": "2x"},
    {"id": 5, "tag": "derivative", "prompt": "At x=3, slope of f(x)=x^2 is:", "choices": ["3", "6", "9", "12"], "answer": "6"},
])

EXPLANATIONS = {
    "algebra_linear": {
        "intuitive": "Linear equations are balance scales. 2x+3=11 → remove 3 from both sides, then divide by 2.",
        "formal": "2x+3=11 ⇒ 2x=8 ⇒ x=4. Linear y=mx+b has constant slope m."
    },
    "rates": {
        "intuitive": "Average rate = rise/run. For x^2 from 2→5: (25−4)/(5−2)=21/3=7.",
        "formal": "Average rate on [a,b] is (f(b)−f(a))/(b−a). For x^2, (25−4)/3=7."
    },
    "derivative": {
        "intuitive": "Derivative = instantaneous slope. Zoom in until the curve looks straight; that slope is the derivative.",
        "formal": "f'(x)=lim_{h→0}(f(x+h)−f(x))/h. For x^2, f'(x)=2x."
    }
}

# ---------------------------
# Time limit (15 min/day)
# ---------------------------
FREE_SECONDS_PER_DAY = 15 * 60

def time_left_today():
    today = str(date.today())
    if "quota" not in st.session_state or today not in st.session_state.quota:
        st.session_state.quota = {today: FREE_SECONDS_PER_DAY}
    return st.session_state.quota[today]

def spend_time(sec):
    today = str(date.today())
    st.session_state.quota[today] = max(0, st.session_state.quota[today] - sec)

# ---------------------------
# Simple learner model
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["qid", "tag", "correct"])

def weakest_tag():
    hist = st.session_state.history
    if hist.empty:
        return np.random.choice(QUESTIONS["tag"].unique())
    stats = hist.groupby("tag")["correct"].mean().sort_values()
    return stats.index[0]

def choose_next_q():
    weak = weakest_tag()
    seen = set(st.session_state.history["qid"].tolist())
    pool = QUESTIONS[~QUESTIONS["id"].isin(seen)]
    weak_pool = pool[pool["tag"] == weak]
    if not weak_pool.empty:
        return weak_pool.sample(1).iloc[0]
    if not pool.empty:
        return pool.sample(1).iloc[0]
    return QUESTIONS.sample(1).iloc[0]

# ---------------------------
# App state machine
# ---------------------------
st.set_page_config(page_title="15-Minute Math (MVP)", page_icon="🧠", layout="centered")
st.title("🧠 15-Minute Math — MVP")
st.caption("Learn your way. Intuitive ↔ Formal. Free with daily 15-min limit.")

premium = st.toggle("Simulate premium (no time limit)", value=False)
remaining = time_left_today()

if not premium and remaining <= 0:
    st.error("You've used your free 15 minutes today. Come back tomorrow or simulate premium above.")
    st.stop()

style = st.radio("Choose your style for explanations:", ["Intuitive", "Formal"], horizontal=True)

if premium:
    st.success("Premium mode simulated — unlimited time.")
else:
    st.info(f"Free time remaining: **{remaining//60}m {remaining%60}s**")

# Initialize per-question state
if "phase" not in st.session_state:
    # phases: "ask" → "feedback"
    st.session_state.phase = "ask"
if "current_q" not in st.session_state:
    st.session_state.current_q = choose_next_q().to_dict()
if "last_feedback" not in st.session_state:
    st.session_state.last_feedback = None

q = st.session_state.current_q
qid = q["id"]

# ----- ASK PHASE -----
if st.session_state.phase == "ask":
    with st.form(key=f"form_q_{qid}", clear_on_submit=False):
        st.subheader("Question")
        st.write(q["prompt"])
        answer = st.radio("Pick one:", q["choices"], index=None, key=f"choice_{qid}")
        submitted = st.form_submit_button("Check answer")
        if submitted:
            if answer is None:
                st.warning("Pick an option first.")
            else:
                correct = (answer == q["answer"])
                # Update history
                st.session_state.history = pd.concat([
                    st.session_state.history,
                    pd.DataFrame([{"qid": qid, "tag": q["tag"], "correct": int(correct)}])
                ], ignore_index=True)

                # Store feedback to show after rerun
                st.session_state.last_feedback = {"correct": correct, "chosen": answer}
                st.session_state.phase = "feedback"

                if not premium:
                    spend_time(8)  # spend time on grading/explaining

                st.rerun()

# ----- FEEDBACK PHASE -----
if st.session_state.phase == "feedback":
    fb = st.session_state.last_feedback or {"correct": False, "chosen": None}
    if fb["correct"]:
        st.success("✅ Correct!")
    else:
        st.error(f"❌ Not quite. Correct answer: **{q['answer']}**")

    # Explanation
    pack = EXPLANATIONS[q["tag"]]
    st.markdown("### Explanation")
    st.write(pack["intuitive" if style == "Intuitive" else "formal"])

    # Interactive bit (for derivative tag)
    if q["tag"] == "derivative":
        st.markdown("#### Interactive intuition: slope on f(x)=x² at a point")
        x0 = st.slider("Pick x₀", min_value=-5.0, max_value=5.0, value=2.0, step=0.1, key=f"x0_{qid}")
        h = 1e-3
        slope = ((x0 + h)**2 - (x0 - h)**2) / (2*h)
        st.write(f"Approximate slope at x₀={x0:.2f} is **{slope:.2f}** (exact 2x ⇒ {2*x0:.2f})")

        xs = np.linspace(-6, 6, 241)
        ys = xs**2
        df = pd.DataFrame({"x": xs, "f(x)=x^2": ys}).set_index("x")
        st.line_chart(df)

        if not premium:
            # Spend time once on interaction view; avoid per-slider tick burns
            if "spent_interactive" not in st.session_state or st.session_state.spent_interactive != qid:
                spend_time(4)
                st.session_state.spent_interactive = qid

    st.markdown("---")
    if st.button("Next question →"):
        # Advance to a brand new question and reset phase
        st.session_state.current_q = choose_next_q().to_dict()
        st.session_state.phase = "ask"
        st.session_state.last_feedback = None
        # Reset prior selection key so radio starts empty for the new q
        st.session_state.pop(f"choice_{qid}", None)
        if not premium:
            spend_time(2)
        st.rerun()

# Progress
if not st.session_state.history.empty:
    st.markdown("### Your current weak area")
    weak = weakest_tag()
    stats = st.session_state.history.groupby("tag")["correct"].mean().round(2)
    st.write(f"**{weak}** is your lowest-accuracy tag so far.")
    st.dataframe(stats.reset_index().rename(columns={"correct": "accuracy"}), use_container_width=True)

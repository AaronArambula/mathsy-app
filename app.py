import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

# ---------------------------
# Minimal "content database"
# ---------------------------
QUESTIONS = pd.DataFrame([
    # Algebra – linear functions
    {"id": 1, "tag": "algebra_linear", "prompt": "Solve for x: 2x + 3 = 11", "choices": ["3", "4", "5", "8"], "answer": "4"},
    {"id": 2, "tag": "algebra_linear", "prompt": "Slope of y = 5x - 7 is:", "choices": ["-7", "0", "5", "12"], "answer": "5"},
    # Functions / rates – bridge to calculus
    {"id": 3, "tag": "rates", "prompt": "Average rate of change of f(x)=x^2 from x=2 to x=5:", "choices": ["3", "7", "9", "11"], "answer": "7"},
    # Derivative basics
    {"id": 4, "tag": "derivative", "prompt": "Derivative of f(x)=x^2 is:", "choices": ["x", "2x", "x^3", "2"], "answer": "2x"},
    {"id": 5, "tag": "derivative", "prompt": "At x=3, slope of f(x)=x^2 is:", "choices": ["3", "6", "9", "12"], "answer": "6"},
])

EXPLANATIONS = {
    "algebra_linear": {
        "intuitive": "Linear equations are balance scales. 2x+3=11 means 'some amount twice, plus 3, equals 11'. Remove 3 from both sides (keep balance), then divide by 2.",
        "formal": "2x+3=11 ⇒ 2x=8 by subtracting 3. Then divide both sides by 2: x=4. A linear function y=mx+b has constant slope m."
    },
    "rates": {
        "intuitive": "Average rate of change is 'rise over run' between two points. For x^2 from 2→5: (25−4)/(5−2)=21/3=7.",
        "formal": "Average rate of change of f on [a,b] is (f(b)−f(a))/(b−a). For f(x)=x^2, a=2, b=5 gives (25−4)/3=7."
    },
    "derivative": {
        "intuitive": "Derivative = instantaneous slope. Zoom into the curve until it looks straight; the slope of that tiny line is the derivative.",
        "formal": "f'(x)=lim_{h→0} (f(x+h)−f(x))/h. For f(x)=x^2, f'(x)=2x."
    }
}

# ---------------------------
# Session / time limit (15 min/day)
# ---------------------------
FREE_SECONDS_PER_DAY = 15 * 60

def minutes_left_today():
    # simple per-browser memory using session_state; good enough for MVP
    today_str = str(date.today())
    if "quota" not in st.session_state:
        st.session_state.quota = {today_str: FREE_SECONDS_PER_DAY}
    else:
        # reset when new day
        if today_str not in st.session_state.quota:
            st.session_state.quota = {today_str: FREE_SECONDS_PER_DAY}
    return st.session_state.quota[today_str]

def burn_seconds(sec):
    today_str = str(date.today())
    st.session_state.quota[today_str] = max(0, st.session_state.quota[today_str] - sec)

# tick a tiny cost for each interaction to simulate usage
INTERACTION_COST_SEC = 6

# ---------------------------
# User model (super simple)
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["qid", "tag", "correct"])

def tag_weakness():
    # Return the tag with lowest accuracy, else random
    hist = st.session_state.history
    if hist.empty:
        return np.random.choice(QUESTIONS["tag"].unique())
    stats = hist.groupby("tag")["correct"].mean().sort_values()
    # If all 1.0, still return something deterministic
    return stats.index[0]

def next_question():
    # Prefer a question from weakest tag not seen recently
    weak = tag_weakness()
    seen = set(st.session_state.history["qid"].tolist())
    candidates = QUESTIONS[~QUESTIONS["id"].isin(seen)]
    # prioritize weak tag
    weak_candidates = candidates[candidates["tag"] == weak]
    if not weak_candidates.empty:
        return weak_candidates.sample(1).iloc[0]
    # else any remaining
    if not candidates.empty:
        return candidates.sample(1).iloc[0]
    # recycle if all seen (MVP simplicity)
    return QUESTIONS.sample(1).iloc[0]

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="15-Minute Math (MVP)", page_icon="🧠", layout="centered")
st.title("🧠 15-Minute Math — MVP")
st.caption("Learn your way. Intuitive ↔ Formal. Free with daily 15-min limit.")

# Fake “premium” toggle for demo
premium = st.toggle("Simulate premium (no time limit)", value=False)

remaining = minutes_left_today()
if not premium and remaining <= 0:
    st.error("You've used your free 15 minutes today. Come back tomorrow or go premium (simulated here).")
    st.stop()

# Learning style
style = st.radio("Choose your style for explanations:", ["Intuitive", "Formal"], horizontal=True)

# Show remaining time
if premium:
    st.success("Premium mode simulated — unlimited time for demo.")
else:
    st.info(f"Free time remaining today: **{remaining//60}m {remaining%60}s**")

# Pull a question
q = next_question()
st.subheader("Question")
st.write(q["prompt"])
answer = st.radio("Pick one:", q["choices"], index=None)

# Submit
submitted = st.button("Check answer")
if submitted:
    is_correct = (answer == q["answer"])
    st.session_state.history = pd.concat([
        st.session_state.history,
        pd.DataFrame([{"qid": q["id"], "tag": q["tag"], "correct": int(is_correct)}])
    ], ignore_index=True)
    if is_correct:
        st.success("✅ Correct!")
    else:
        st.error(f"❌ Not quite. Correct answer: **{q['answer']}**")

    if not premium:
        burn_seconds(INTERACTION_COST_SEC)

    # Explanation block
    exp = EXPLANATIONS[q["tag"]]
    st.markdown("### Explanation")
    st.write(exp["intuitive" if style == "Intuitive" else "formal"])

    # Tiny interactive for derivatives (prove the “intuitive” vibe)
    if q["tag"] == "derivative":
        st.markdown("#### Interactive intuition: slope on f(x)=x² at a point")
        x0 = st.slider("Pick x₀", min_value=-5.0, max_value=5.0, value=2.0, step=0.1)
        # approximate slope via symmetric difference
        h = 1e-3
        slope = ((x0 + h)**2 - (x0 - h)**2) / (2*h)
        st.write(f"Approximate slope at x₀={x0:.2f} is **{slope:.2f}** (exact derivative 2x ⇒ {2*x0:.2f})")

        xs = np.linspace(-6, 6, 241)
        ys = xs**2
        df = pd.DataFrame({"x": xs, "f(x)=x^2": ys}).set_index("x")
        st.line_chart(df)

        if not premium:
            burn_seconds(INTERACTION_COST_SEC)

# Progress & weak areas
if not st.session_state.history.empty:
    st.markdown("---")
    st.markdown("### Your current weak area")
    weak = tag_weakness()
    tag_stats = st.session_state.history.groupby("tag")["correct"].mean().round(2)
    st.write(f"**{weak}** is currently your lowest-accuracy tag.")
    st.dataframe(tag_stats.reset_index().rename(columns={"correct": "accuracy"}), use_container_width=True)

    if not premium:
        burn_seconds(2)

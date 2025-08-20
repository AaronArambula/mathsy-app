import streamlit as st
import pandas as pd
import random

# --- Load SAT question bank ---
@st.cache_data
def load_questions():
    df = pd.read_csv("sat_bank.csv")
    return df

questions_df = load_questions()

# --- Initialize session state ---
if "current_q" not in st.session_state:
    st.session_state.current_q = None
if "history" not in st.session_state:
    st.session_state.history = []

def pick_question():
    return questions_df.sample(1).iloc[0]

# --- Render question ---
def render_question(q):
    st.subheader(q["question"])
    if not pd.isna(q.get("choice_A", None)):
        choices = [q.get(f"choice_{c}", None) for c in ["A", "B", "C", "D"] if pd.notna(q.get(f"choice_{c}", None))]
        answer = st.radio("Choose your answer:", choices, index=None, key="user_answer")
    else:
        answer = st.text_input("Enter your answer:", key="user_answer")
    return answer

# --- Check answer ---
def check_answer(user_answer, correct_answer, explanation):
    if str(user_answer).strip().lower() == str(correct_answer).strip().lower():
        st.success("✅ Correct!")
        result = True
    else:
        st.error(f"❌ Incorrect. Correct answer: {correct_answer}")
        result = False

    with st.expander("See explanation"):
        st.markdown(explanation)

    # Step-by-step mode
    if ">>" in explanation:
        steps = explanation.split(">>")
        st.markdown("### Step-by-Step Walkthrough")
        for i, step in enumerate(steps, 1):
            if st.button(f"Step {i}", key=f"step_{i}"):
                st.info(step.strip())

    return result

# --- Main flow ---
st.title("📘 SAT Practice MVP")

if st.button("New Question"):
    st.session_state.current_q = pick_question()

if st.session_state.current_q is not None:
    q = st.session_state.current_q
    user_answer = render_question(q)

    if st.button("Check Answer"):
        correct = check_answer(user_answer, q["answer"], q.get("explanation", "No explanation available."))
        st.session_state.history.append({"id": q["id"], "tag": q.get("tag", "General"), "correct": correct})
        st.session_state.current_q = None

# --- Stats ---
if len(st.session_state.history) > 0:
    st.markdown("## 📊 Your Progress")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df)
    tag_stats = hist_df.groupby("tag")["correct"].mean().round(2) * 100
    st.bar_chart(tag_stats)

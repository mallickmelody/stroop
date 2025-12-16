import streamlit as st
import random
import time
import pandas as pd
import numpy as np
import pickle

# ---------------- Load ML model ----------------
model = pickle.load(open("stroop_model.pkl", "rb"))

# ---------------- Stroop Config ----------------
COLOR_NAMES = ["RED", "GREEN", "BLUE", "YELLOW"]
COLOR_HEX = {
    "RED": "#d62828",
    "GREEN": "#2a9d8f",
    "BLUE": "#0077b6",
    "YELLOW": "#f4d35e"
}
NUM_QUESTIONS = 20
ISI = 0.3  # inter-trial interval in seconds

# ---------------- Helper Functions ----------------
def make_trial():
    word = random.choice(COLOR_NAMES)
    ink = random.choice(COLOR_NAMES)
    return {"word": word, "ink": ink}

def show_stimulus(trial):
    st.markdown(
        f"<div style='text-align:center; margin-top:50px;'>"
        f"<span style='font-size:100px; font-weight:700; color:{COLOR_HEX[trial['ink']]};'>{trial['word']}</span>"
        f"</div>", unsafe_allow_html=True
    )

def record_response(trial, response, rt):
    correct = int(response.upper() == trial["ink"][0])
    st.session_state.results.append({
        "trial": st.session_state.current_idx + 1,
        "word": trial["word"],
        "ink": trial["ink"],
        "response": response.upper(),
        "correct": correct,
        "reaction_time_s": rt
    })

def reset_session():
    st.session_state.stage = "instructions"
    st.session_state.trials = [make_trial() for _ in range(NUM_QUESTIONS)]
    st.session_state.current_idx = 0
    st.session_state.results = []
    st.session_state.start_time = None
    st.session_state.user_age = None
    st.session_state.test_finished = False

# ---------------- Session State Init ----------------
if "stage" not in st.session_state:
    reset_session()

st.set_page_config(page_title="Stroop Test", layout="centered")

# ---------------- UI ----------------
st.title("🧠 Stroop Cognitive Test")
st.write("Select the **ink color** of the word as fast and accurately as possible.")

# ---------------- Instructions Screen ----------------
if st.session_state.stage == "instructions":
    st.header("Instructions")
    st.markdown("""
        - You will see a color word displayed in colored ink.
        - Select the **color of the ink**, not the text.
        - Try to respond quickly and accurately.
        - The test consists of 20 trials.
    """)

    # Age input
    st.session_state.user_age = st.number_input(
        "Enter your age (years):", min_value=10, max_value=120, step=1
    )

    if st.button("Start Test") and st.session_state.user_age is not None:
        st.session_state.stage = "test"
        st.session_state.current_idx = 0
        st.session_state.results = []
        st.session_state.start_time = time.time()

# ---------------- Test Screen ----------------
elif st.session_state.stage == "test":
    idx = st.session_state.current_idx

    if idx >= NUM_QUESTIONS:
        st.session_state.stage = "results"
        st.session_state.test_finished = True
    else:
        trial = st.session_state.trials[idx]
        st.write(f"Trial {idx+1} / {NUM_QUESTIONS}")
        show_stimulus(trial)

        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()

        # Color buttons
        cols = st.columns(len(COLOR_NAMES))
        clicked_color = None
        for i, color in enumerate(COLOR_NAMES):
            if cols[i].button(color):
                clicked_color = color[0]

        # Record response
        if clicked_color:
            rt = time.time() - st.session_state.start_time
            record_response(trial, clicked_color, rt)
            st.session_state.current_idx += 1
            st.session_state.start_time = time.time()
            time.sleep(ISI)  # short interval before next trial
            st.experimental_rerun()

# ---------------- Results Screen ----------------
elif st.session_state.stage == "results" and st.session_state.test_finished:
    st.header("Results")
    df = pd.DataFrame(st.session_state.results)

    if df.empty:
        st.info("No data collected.")
    else:
        # Compute congruent/incongruent RTs
        df["congruent"] = df.apply(lambda x: x["word"] == x["ink"], axis=1)
        mean_cong = df[df["congruent"]]["reaction_time_s"].mean()
        mean_incong = df[~df["congruent"]]["reaction_time_s"].mean()
        interference = mean_incong - mean_cong
        errors = (~df["correct"].astype(bool)).sum()

        st.write("### Stroop Test Summary")
        st.metric("Mean Congruent RT (s)", f"{mean_cong:.3f}")
        st.metric("Mean Incongruent RT (s)", f"{mean_incong:.3f}")
        st.metric("Interference (s)", f"{interference:.3f}")
        st.metric("Errors", errors)

        # ML prediction
        X_user = np.array([[st.session_state.user_age, mean_cong, mean_incong, interference, errors]])
        zone_pred = model.predict(X_user)[0]
        prob = model.predict_proba(X_user)[0]
        red_prob = prob[list(model.classes_).index("red")] * 100

        st.write("---")
        st.write("## Predicted Zone")
        st.write(f"Zone: {zone_pred}")
        st.write(f"Red Zone Probability: {red_prob:.1f}%")

        if zone_pred == "blue":
            st.success("BLUE ZONE — Superior executive control")
        elif zone_pred == "green":
            st.info("GREEN ZONE — Age-appropriate cognition")
        else:
            st.error("RED ZONE — Elevated cognitive risk")

        # Detailed trial-level table
        st.write("### Trial-level Data")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results CSV", data=csv, file_name="stroop_results.csv", mime="text/csv")

    if st.button("Restart Test"):
        reset_session()
        st.experimental_rerun()

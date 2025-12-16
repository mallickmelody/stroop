import streamlit as st
import random
import time
import numpy as np
import pickle
import pandas as pd

# ------------------ LOAD MODEL PIPELINE ------------------
pipeline = pickle.load(open("stroop_model.pkl", "rb"))

# ------------------ CONFIG ------------------
COLORS = ["RED", "GREEN", "BLUE"]
COLOR_MAP = {"RED": "red", "GREEN": "green", "BLUE": "blue"}
TOTAL_TRIALS = 20

# ------------------ HELPERS ------------------
def reset_test():
    st.session_state.started = False
    st.session_state.trial = 0
    st.session_state.age = 0
    st.session_state.trials_data = []  # List of dicts: store each trial
    st.session_state.start_time = 0
    st.session_state.test_finished = False

def next_trial():
    st.session_state.trial += 1
    st.session_state.start_time = time.time()

# ------------------ SESSION STATE ------------------
if "started" not in st.session_state:
    reset_test()

# ------------------ FIRST SCREEN: INSTRUCTIONS ------------------
if not st.session_state.started:
    st.title("Stroop Cognitive Assessment")
    st.subheader("Instructions:")
    st.markdown("""
    1. Select your age using the slider.  
    2. Click **Start Test** to begin.  
    3. You will see 20 words, **one at a time**.  
    4. Select the **INK color** of the word (not the word itself).  
    5. Your reaction time, correctness, and interference will be recorded.  
    6. At the end, your results will be displayed.  
    """)
    st.session_state.age = st.slider("Select your age", 18, 80, 55)

    if st.button("Start Test"):
        st.session_state.started = True
        st.session_state.trial = 0
        st.session_state.start_time = time.time()
        st.session_state.trials_data = []
        st.session_state.test_finished = False

# ------------------ TRIAL SCREEN ------------------
elif st.session_state.started and st.session_state.trial < TOTAL_TRIALS:
    st.write(f"Trial {st.session_state.trial + 1} of {TOTAL_TRIALS}")
    st.progress((st.session_state.trial + 1)/TOTAL_TRIALS)

    # Pick trial type
    is_congruent = random.choice([True, False])
    if is_congruent:
        word = ink = random.choice(COLORS)
    else:
        word = random.choice(COLORS)
        ink = random.choice([c for c in COLORS if c != word])

    # Show word
    st.markdown(f"<h1 style='color:{COLOR_MAP[ink]}; font-size: 80px'>{word}</h1>", unsafe_allow_html=True)

    # User response
    choice = st.radio("Select the INK color:", COLORS, key=f"trial_{st.session_state.trial}")

    if st.button("Submit", key=f"submit_{st.session_state.trial}"):
        rt = (time.time() - st.session_state.start_time) * 1000  # ms
        correct = choice == ink

        # Save trial data
        st.session_state.trials_data.append({
            "trial": st.session_state.trial + 1,
            "word": word,
            "ink": ink,
            "choice": choice,
            "correct": correct,
            "reaction_time_ms": rt
        })

        next_trial()

# ------------------ RESULTS SCREEN ------------------
elif st.session_state.started and st.session_state.trial >= TOTAL_TRIALS and not st.session_state.test_finished:
    st.subheader("Stroop Test Completed! Here are your results:")

    df_results = pd.DataFrame(st.session_state.trials_data)

    # Display per-trial results
    st.dataframe(df_results)

    # Compute summary stats
    mean_cong = df_results[df_results["word"] == df_results["ink"]]["reaction_time_ms"].mean()
    mean_incong = df_results[df_results["word"] != df_results["ink"]]["reaction_time_ms"].mean()
    interference = mean_incong - mean_cong
    errors = (~df_results["correct"]).sum()

    st.write(f"Mean Congruent RT: {mean_cong:.1f} ms")
    st.write(f"Mean Incongruent RT: {mean_incong:.1f} ms")
    st.write(f"Interference: {interference:.1f} ms")
    st.write(f"Total Errors: {errors}")

    # Predict zone
    X_user = np.array([[st.session_state.age, mean_cong, mean_incong, interference, errors]])
    zone_pred = pipeline.predict(X_user)[0]
    prob = pipeline.predict_proba(X_user)[0]
    red_prob = prob[list(pipeline.classes_).index("red")] * 100

    st.write(f"Predicted Zone: {zone_pred}")
    st.write(f"Red Zone Probability: {red_prob:.1f}%")

    if zone_pred == "blue":
        st.success("BLUE ZONE — Superior executive control")
    elif zone_pred == "green":
        st.info("GREEN ZONE — Age-appropriate cognition")
    else:
        st.error("RED ZONE — Elevated cognitive risk")

    # Mark test finished
    st.session_state.test_finished = True

    if st.button("Restart Test"):
        reset_test()

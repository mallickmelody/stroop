import streamlit as st
import random
import time
import numpy as np
import pickle

# ------------------ LOAD MODEL PIPELINE ------------------
pipeline = pickle.load(open("stroop_model.pkl", "rb"))

# ------------------ CONFIG ------------------
COLORS = ["RED", "GREEN", "BLUE"]
COLOR_MAP = {"RED": "red", "GREEN": "green", "BLUE": "blue"}

# ------------------ HELPERS ------------------
def reset_test():
    st.session_state.started = False
    st.session_state.trial = 0
    st.session_state.start_time = 0
    st.session_state.cong_rt = []
    st.session_state.incong_rt = []
    st.session_state.errors = 0

# ------------------ SESSION STATE ------------------
if "started" not in st.session_state:
    reset_test()

# ------------------ UI ------------------
st.title("Stroop Cognitive Assessment")

age = st.slider("Select your age", 18, 80, 55)

if not st.session_state.started:
    if st.button("Start Stroop Test"):
        st.session_state.started = True
        st.session_state.trial = 0
        st.session_state.cong_rt = []
        st.session_state.incong_rt = []
        st.session_state.errors = 0
        st.session_state.start_time = time.time()

# ------------------ STROOP TRIALS ------------------
if st.session_state.started and st.session_state.trial < 20:

    is_congruent = random.choice([True, False])

    if is_congruent:
        word = ink = random.choice(COLORS)
    else:
        word = random.choice(COLORS)
        ink = random.choice([c for c in COLORS if c != word])

    st.markdown(
        f"<h1 style='color:{COLOR_MAP[ink]}; font-size: 80px'>{word}</h1>",
        unsafe_allow_html=True
    )

    choice = st.radio("Select the INK color:", COLORS, key=st.session_state.trial)

    if st.button("Submit"):
        rt = (time.time() - st.session_state.start_time) * 1000  # ms

        if choice != ink:
            st.session_state.errors += 1

        if is_congruent:
            st.session_state.cong_rt.append(rt)
        else:
            st.session_state.incong_rt.append(rt)

        st.session_state.trial += 1
        st.session_state.start_time = time.time()
        st.experimental_rerun()

# ------------------ RESULTS ------------------
elif st.session_state.started and st.session_state.trial == 20:

    mean_cong = np.mean(st.session_state.cong_rt)
    mean_incong = np.mean(st.session_state.incong_rt)
    interference = mean_incong - mean_cong
    errors = st.session_state.errors

    X_user = np.array([[age, mean_cong, mean_incong, interference, errors]])

    zone_pred = pipeline.predict(X_user)[0]
    prob = pipeline.predict_proba(X_user)[0]
    red_prob = prob[list(pipeline.classes_).index("red")] * 100

    st.subheader("Stroop Test Results")
    st.write(f"Congruent RT: {mean_cong:.1f} ms")
    st.write(f"Incongruent RT: {mean_incong:.1f} ms")
    st.write(f"Interference: {interference:.1f} ms")
    st.write(f"Errors: {errors}")
    st.write(f"Predicted Zone: {zone_pred}")
    st.write(f"Red Zone Probability: {red_prob:.1f}%")

    if zone_pred == "blue":
        st.success("BLUE ZONE — Superior executive control")
    elif zone_pred == "green":
        st.info("GREEN ZONE — Age-appropriate cognition")
    else:
        st.error("RED ZONE — Elevated cognitive risk")

    if st.button("Restart Test"):
        reset_test()
        st.experimental_rerun()

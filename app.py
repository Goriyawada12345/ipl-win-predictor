import streamlit as st
import pickle
import numpy as np

# Load model and encoders
model = pickle.load(open("ipl_model.pkl", "rb"))
encoders = pickle.load(open("label_encoders.pkl", "rb"))

# Page config
st.set_page_config(page_title="🏏 IPL Win Predictor", layout="centered")

# Title and Description
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🏆 IPL Win Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Predict the chances of a team winning based on match status</h4>", unsafe_allow_html=True)
st.markdown("---")

# Dropdown options from encoders
teams = encoders['batting_team'].classes_
venues = encoders['venue'].classes_

# Layout
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox("🏏 Batting Team", teams)
        current_score = st.number_input("🎯 Current Score", min_value=0, max_value=300, value=75)
        overs = st.number_input("⏱️ Overs Completed", min_value=0.0, max_value=20.0, step=0.1, value=10.0)
        target = st.number_input("🏁 Target Score", min_value=1, max_value=300, value=160)

    with col2:
        bowling_team = st.selectbox("🎽 Bowling Team", teams)
        wickets = st.number_input("💥 Wickets Out", min_value=0, max_value=10, value=3)
        venue = st.selectbox("📍 Match Venue", venues)

    st.markdown("---")
    submit = st.form_submit_button("🔮 Predict Win Probability")

# Prediction logic
if submit:
    try:
        # Derived features
        balls_left = int(120 - overs * 6)

        # Encoded features
        encoded_batting = encoders['batting_team'].transform([batting_team])[0]
        encoded_bowling = encoders['bowling_team'].transform([bowling_team])[0]
        encoded_venue = encoders['venue'].transform([venue])[0]

        # Input array (must match training feature order)
        input_features = np.array([[encoded_batting, encoded_bowling, encoded_venue,
                                    current_score, wickets, overs, target, balls_left]])

        # Model prediction
        probabilities = model.predict_proba(input_features)[0]
        loss_prob = probabilities[0] * 100
        win_prob = probabilities[1] * 100

        # Display results
        st.markdown("### 🧮 Prediction Result")
        st.success(f"✅ **{batting_team} Win Probability: {win_prob:.2f}%**")
        st.error(f"❌ **{bowling_team} Win Probability: {loss_prob:.2f}%**")

        # Optional progress bar
        st.progress(int(win_prob))

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

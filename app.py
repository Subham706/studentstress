import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("student_stress_model.pkl")

# Features (same order as training)
feature_names = [
    'anxiety_level', 'self_esteem', 'mental_health_history',
    'depression', 'headache', 'blood_pressure', 'sleep_quality',
    'breathing_problem', 'noise_level', 'living_conditions', 'safety',
    'basic_needs', 'academic_performance', 'study_load',
    'teacher_student_relationship', 'future_career_concerns',
    'social_support', 'peer_pressure', 'extracurricular_activities',
    'bullying'
]

# Mapping for stress levels
stress_mapping = {
    0: "Low Stress",
    1: "Moderate Stress",
    2: "High Stress"
}

st.title("🎓 Student Stress Level Prediction")

st.write("Answer **Yes** or **No** to the following questions:")

# User inputs
user_values = []
for feature in feature_names:
    answer = st.radio(
        f"Does the student have {feature.replace('_', ' ')}?",
        ("Yes", "No"),
        index=1
    )
    user_values.append(1 if answer == "Yes" else 0)

# Predict button
if st.button("Predict Stress Level"):
    user_df = pd.DataFrame([user_values], columns=feature_names)
    prediction = model.predict(user_df)[0]
    st.subheader(f"Predicted Stress Level: **{stress_mapping.get(prediction, 'Unknown')}**")

# app.py – Streamlit web app for Student Risk Prediction

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load trained pipeline
pipeline = joblib.load("models/student_risk_pipeline.joblib")

st.title("Student Performance Risk Prediction")
st.write("Early-warning style app: predict if a student is at risk of failing.")

# --- Input form ---

# Categorical inputs (must match training feature names and categories)
school = st.selectbox("School", ["GP", "MS"])
sex = st.selectbox("Sex", ["F", "M"])
address = st.selectbox("Home address", ["U", "R"])
famsize = st.selectbox("Family size", ["LE3", "GT3"])
Pstatus = st.selectbox("Parents' status", ["T", "A"])
schoolsup = st.selectbox("Extra school support", ["yes", "no"])
famsup = st.selectbox("Family support", ["yes", "no"])
activities = st.selectbox("Extra-curricular activities", ["yes", "no"])
nursery = st.selectbox("Attended nursery", ["yes", "no"])
higher = st.selectbox("Wants higher education", ["yes", "no"])
internet = st.selectbox("Internet at home", ["yes", "no"])
romantic = st.selectbox("In a romantic relationship", ["yes", "no"])

# Numeric inputs (use same ranges as dataset)
age = st.slider("Age", 15, 22, 17)
studytime = st.slider("Weekly study time (1–4)", 1, 4, 2)
failures = st.slider("Past class failures (0–4)", 0, 4, 0)
absences = st.slider("Number of absences", 0, 50, 2)
traveltime = st.slider("Travel time (1–4)", 1, 4, 1)
freetime = st.slider("Free time (1–5)", 1, 5, 3)
goout = st.slider("Going out (1–5)", 1, 5, 3)
Dalc = st.slider("Workday alcohol (1–5)", 1, 5, 1)
Walc = st.slider("Weekend alcohol (1–5)", 1, 5, 1)
health = st.slider("Current health (1–5)", 1, 5, 3)

if st.button("Predict risk"):
    # Build a single-row DataFrame with same columns as training
    input_dict = {
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "age": age,
        "traveltime": traveltime,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "school": school,
        "sex": sex,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
    }
    input_df = pd.DataFrame([input_dict])

    # Get prediction and probability
    risk_pred = pipeline.predict(input_df)[0]          # 0 or 1
    risk_proba = pipeline.predict_proba(input_df)[0,1] # P(fail)

    st.write("---")
    st.subheader("Prediction")

    if risk_pred == 1:
        st.error(f"⚠️ Student is predicted **AT RISK** of failing.\n\nEstimated failure probability: **{risk_proba:.2f}**")
    else:
        st.success(f"✅ Student is predicted **NOT at risk**.\n\nEstimated failure probability: **{risk_proba:.2f}**")

    st.caption("Note: This is a statistical model; educators should combine it with qualitative judgement.")

import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/addiction_model.pkl")

st.title("📱 Mobile Addiction Risk Predictor")

screen = st.slider("Screen Time", 1, 15)
sleep = st.slider("Sleep Hours", 1, 12)
social = st.slider("Social Media Hours", 0, 10)
gaming = st.slider("Gaming Hours", 0, 10)
pickups = st.slider("Phone Pickups", 1, 200)
study = st.slider("Study Hours", 0, 12)
work = st.slider("Work Hours", 0, 12)

if st.button("Predict Risk"):
    data = pd.DataFrame([[screen, sleep, social, gaming, pickups, study, work]],
                        columns=[
                            "Screen_Time",
                            "Sleep_Hours",
                            "Social_Media",
                            "Gaming_Hours",
                            "Pickups",
                            "Study_Hours",
                            "Work_Hours"
                        ])

    pred = model.predict(data)[0]

    if pred == 0:
        st.success("🟢 Low Risk")
    elif pred == 1:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")
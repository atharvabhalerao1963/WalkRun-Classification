import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("final_xgboost_model.pkl")

# App title
st.title("🏃 Walking vs Running Classification App")

st.markdown("### Enter Sensor Values Below:")

# Input fields
wrist = st.selectbox("Wrist (0 = Left, 1 = Right)", [0, 1])

acc_x = st.number_input("Acceleration X", value=0.0)
acc_y = st.number_input("Acceleration Y", value=0.0)
acc_z = st.number_input("Acceleration Z", value=0.0)

gyro_x = st.number_input("Gyro X", value=0.0)
gyro_y = st.number_input("Gyro Y", value=0.0)
gyro_z = st.number_input("Gyro Z", value=0.0)

# Feature Engineering (IMPORTANT — same as training)
acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

# Predict button
if st.button("Predict Activity"):

    input_data = np.array([[wrist, acc_x, acc_y, acc_z,
                            gyro_x, gyro_y, gyro_z,
                            acc_mag, gyro_mag]])

    prediction = model.predict(input_data)

    # Output
    if prediction[0] == 0:
        st.success("🚶 The person is WALKING")
    else:
        st.success("🏃 The person is RUNNING")
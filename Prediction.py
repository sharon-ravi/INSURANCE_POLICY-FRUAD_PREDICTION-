import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import load_model # type: ignore

# Load the models
fraud_model = load_model("lstm_fraud_model.h5")
policy_model = load_model('policy_model.h5')

# Load the dataset and fit the scalers
data = pd.read_csv('final .csv')
fraud_features = ['Age', 'ClaimAmount', 'PastNumberOfClaims', 'DriverRating', 'Deductible']
policy_features = ['WeekOfMonthClaimed', 'DayOfWeekClaimed', 'MonthClaimed', 'AgeOfPolicyHolder', 
                   'ClaimAmount', 'AgeOfVehicle', 'Year']

# Scaling for fraud detection features
fraud_scaler = StandardScaler()
fraud_scaler.fit(data[fraud_features])

# Scaling for policy prediction features
policy_scaler = StandardScaler()
policy_scaler.fit(data[policy_features])

# Label encoding for policy prediction
label_encoder = LabelEncoder()
data['PolicyType'] = label_encoder.fit_transform(data['PolicyType'])

# Custom CSS for buttons and layout
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color:white;
        height: 3em;
        width: 12em;
        font-size: 18px;
        border-radius:10px;
        border:2px solid #FFFFFF;
    }
    div.stButton > button:hover {
        background-color: #45a049;
        color:white;
    }
    .prediction-result {
        color: #FF4500;
        font-size: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Function to get fraud detection user input
def get_fraud_input():
    st.markdown('<h3 style="color:#00CED1;">Please provide the following details for Fraud Detection:</h3>', unsafe_allow_html=True)
    user_input = {}
    user_input['Age'] = st.number_input("Enter value for Age:", min_value=0, max_value=100, value=25)
    user_input['ClaimAmount'] = st.number_input("Enter value for ClaimAmount:", min_value=0, value=10000)
    user_input['PastNumberOfClaims'] = st.number_input("Enter value for PastNumberOfClaims:", min_value=0, max_value=100, value=2)
    user_input['DriverRating'] = st.number_input("Enter value for DriverRating:", min_value=1, max_value=5, value=3)
    user_input['Deductible'] = st.number_input("Enter value for Deductible:", min_value=0, value=500)
    return pd.DataFrame([user_input])

# Function to get insurance policy input
def get_policy_input():
    st.markdown('<h3 style="color:#4682B4;">Enter the details for Policy Prediction:</h3>', unsafe_allow_html=True)
    week_of_month = st.number_input("Week of the Month of the Claim:", min_value=1, max_value=5)
    day_of_week = st.number_input("Day of the Week of the Claim:", min_value=1, max_value=7)
    month_of_year = st.number_input("Month of the Year of the Claim:", min_value=1, max_value=12)
    age = st.number_input("Age of the Policy Holder:", min_value=0, max_value=100, value=30)
    claim_amount = st.number_input("Claim Amount:", min_value=0, value=250000)
    vehicle_age = st.number_input("Age of the Vehicle:", min_value=0, value=5)
    claim_year = st.number_input("Year of the Claim:", min_value=1990, max_value=2024, value=2024)
    return np.array([week_of_month, day_of_week, month_of_year, age, claim_amount, vehicle_age, claim_year]).reshape(1, -1)

# Home page with two buttons
if st.session_state.page == 'home':
    st.markdown('<h1 style="color: #FF6347;">Insurance Fraud and Policy Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #4682B4;">Select an option:</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.5])
    with col1:
        if st.button("Fraud Detection"):
            st.session_state.page = 'fraud_detection'
    with col2:
        if st.button("Insurance Policy"):
            st.session_state.page = 'insurance_policy'

# Fraud Detection page
if st.session_state.page == 'fraud_detection':
    st.markdown('<h2 style="color: #FF8C00;">Fraud Detection</h2>', unsafe_allow_html=True)
    
    # Collect user input
    fraud_input = get_fraud_input()

    # Button to trigger prediction
    if st.button("Detect Fraud"):
        # Scale the input
        fraud_scaled_input = fraud_scaler.transform(fraud_input[fraud_features])

        # Reshape input for the LSTM model
        fraud_input_reshaped = fraud_scaled_input.reshape(1, 1, len(fraud_features))

        # Predict using the trained LSTM model
        fraud_prediction = fraud_model.predict(fraud_input_reshaped)
        st.markdown(f"<p class='prediction-result'>Raw Prediction Probability of Fraud: <strong>{fraud_prediction[0][0]:.4f}</strong></p>", unsafe_allow_html=True)
        
        # Custom decision-making based on input values (optional)
        if fraud_input['ClaimAmount'].values[0] > 25000 and fraud_input['PastNumberOfClaims'].values[0] > 5:
            decision = "Fraud"
        else:
            decision = "Not Fraud"
        
        st.markdown(f"<p class='prediction-result'>Decision: <strong>{decision}</strong></p>", unsafe_allow_html=True)

    # Back button to return to home
    if st.button("Back"):
        st.session_state.page = 'home'

# Insurance Policy Prediction page
if st.session_state.page == 'insurance_policy':
    st.markdown('<h2 style="color:#FF6347;">Insurance Policy Prediction</h2>', unsafe_allow_html=True)
    
    # Collect user input
    policy_input = get_policy_input()

    # Button to trigger prediction
    if st.button("Predict Policy"):
        # Scale the input
        policy_input_scaled = policy_scaler.transform(policy_input)

        # Predict using the trained model
        policy_prediction_proba = policy_model.predict(policy_input_scaled)
        predicted_class = np.argmax(policy_prediction_proba)
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        
        st.markdown(f"<p class='prediction-result'>Predicted Policy Type: <strong>{predicted_label}</strong></p>", unsafe_allow_html=True)

    # Back button to return to home
    if st.button("Back"):
        st.session_state.page = 'home'

import os
import streamlit as st
import pandas as pd
import joblib

# Download and load the model from Hugging Face Hub
model_path = os.path.join(os.path.dirname(__file__), "tourism_project_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Insurance Charges Prediction
st.title("Tourism Purchase Prediction App")
st.write("""
This application predicts the **Tourism Purchase** based on personal and lifestyle details.
Please enter the required information below to get a prediction.
""")

# User input
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
typeofContact = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])
cityTier = st.selectbox("CityTier", ["1", "2","3"])
durationOfPitch = st.number_input("DurationOfPitch", min_value=1, max_value=200, value=10, step=1)
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer","Small Business","Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
numberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=10, value=3, step=1)
numberOfFollowups = st.number_input("NumberOfFollowups", min_value=1, max_value=10, value=3, step=1)
productPitched = st.selectbox("ProductPitched", ["Deluxe", "Basic","Standard","Super Deluxe","King"])
preferredPropertyStar = st.number_input("PreferredPropertyStar", min_value=1, max_value=5, value=3, step=1)
maritalStatus = st.selectbox("MaritalStatus", ["Single","Married","Unmarried","Divorced"])
numberOfTrips = st.number_input("NumberOfTrips", min_value=1, max_value=30, value=1, step=1)
passport = st.selectbox("Passport", ["yes", "no"])
pitchSatisfactionScore = st.number_input("PitchSatisfactionScore", min_value=1, max_value=5, value=3, step=1)
ownCar = st.selectbox("OwnCar", ["yes", "no"])
numberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=5, value=0, step=1)
designation = st.selectbox("Designation", ["Manager","Executive","Senior Manager","VP","AVP"])
monthlyIncome = st.number_input("MonthlyIncome", min_value=1, max_value=1000000, value=23000, step=1000)

# Helper function to convert yes/no to binary
def to_binary(value):
    return 1 if value == "yes" else 0

def binary_to_text(value):
    return "Yes" if value == 1 else "No"
    
# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age ,
    'TypeofContact': typeofContact ,
    'CityTier': cityTier ,
    'DurationOfPitch': durationOfPitch ,
    'Occupation': occupation ,
    'Gender': gender ,
    'NumberOfPersonVisiting': numberOfPersonVisiting ,
    'NumberOfFollowups': numberOfFollowups ,
    'ProductPitched': productPitched ,
    'PreferredPropertyStar': preferredPropertyStar ,
    'MaritalStatus': maritalStatus ,
    'NumberOfTrips': numberOfTrips ,
    'Passport': to_binary(passport),  # Convert to binary
    'PitchSatisfactionScore': pitchSatisfactionScore ,
    'OwnCar': to_binary(ownCar),  # Convert to binary
    'NumberOfChildrenVisiting': numberOfChildrenVisiting ,
    'Designation': designation ,
    'MonthlyIncome': monthlyIncome
}])

# Prediction
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Estimated Purchase Product Taken Decision: **: {binary_to_text(prediction)} **")

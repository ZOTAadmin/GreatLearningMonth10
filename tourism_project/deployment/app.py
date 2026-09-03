import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="ZOTAadmin/GreatLearningMonth10",
    filename="tourism_project_v1.joblib"
)
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

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'age': age ,
    'typeofContact': typeofContact ,
    'cityTier': cityTier ,
    'durationOfPitch': durationOfPitch ,
    'occupation': occupation ,
    'gender': gender ,
    'numberOfPersonVisiting': numberOfPersonVisiting ,
    'numberOfFollowups': numberOfFollowups ,
    'productPitched': productPitched ,
    'preferredPropertyStar': preferredPropertyStar ,
    'maritalStatus': maritalStatus ,
    'numberOfTrips': numberOfTrips ,
    'passport': passport ,
    'pitchSatisfactionScore': pitchSatisfactionScore ,
    'ownCar': ownCar ,
    'numberOfChildrenVisiting': numberOfChildrenVisiting ,
    'designation': designation ,
    'monthlyIncome': monthlyIncome
}])

# Prediction
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Estimated Purchase Product Taken Decision: **${prediction:,.2f}**")

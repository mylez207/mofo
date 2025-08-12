import streamlit as st
import pandas as pd
import joblib  # use joblib, not pickle

# Load the trained pipeline
model = joblib.load("best_model.pkl")

st.title("🏠 House Price Prediction App")
st.write("This app predicts house prices based on input features.")

# Input fields
square_feet = st.number_input("Square Feet", min_value=500, max_value=10000, value=2000)
num_bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
num_bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
location = st.text_input("Location", "Downtown")

if st.button("Predict Price"):
    # Prepare input dataframe matching expected columns of your pipeline
    input_data = pd.DataFrame({
        "square_feet": [square_feet],
        "num_bedrooms": [num_bedrooms],
        "num_bathrooms": [num_bathrooms],
        "location": [location]
    })
    
    # Predict
    prediction = model.predict(input_data)
    
    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")

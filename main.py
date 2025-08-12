import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
model = joblib.load("best_model.pkl")

st.title("🏠 House Price Prediction App")
st.write("This app predicts house prices based on input features.")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    # Basic house features
    st.subheader("🏗️ House Details")
    Area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=2000)
    Bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    Bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    Total_Rooms = st.number_input("Total Rooms", min_value=3, max_value=20, value=6)
    Floors = st.number_input("Floors", min_value=1, max_value=5, value=2)

with col2:
    # Additional features
    st.subheader("🏡 Property Features")
    YearBuilt = st.number_input("Year Built", min_value=1900, max_value=2024, value=2000)
    House_Age = st.number_input("House Age (years)", min_value=0, max_value=100, value=24)
    
    Location = st.selectbox("Location", 
                           options=["Urban","Rural"], 
                           index=1)
    
    Condition = st.selectbox("Condition", 
                            options=["Excellent", "Good", "Fair", "Poor"], 
                            index=1)
    
    Garage = st.selectbox("Garage", 
                         options=["Yes", "No"], 
                         index=0)

if st.button("🔮 Predict Price"):
    # Create input data with ALL required columns
    input_data = pd.DataFrame({
        "Area": [Area],
        "Bedrooms": [Bedrooms], 
        "Bathrooms": [Bathrooms],
        "Total_Rooms": [Total_Rooms],
        "Floors": [Floors],
        "YearBuilt": [YearBuilt],
        "House_Age": [House_Age],
        "Location": [Location],
        "Condition": [Condition],
        "Garage": [Garage]
    })
    
    # Show the input data for verification
    with st.expander("📊 View Input Data"):
        st.write(input_data)
    
    try:
        prediction = model.predict(input_data)
        st.success(f"💰 **Estimated House Price: ${prediction[0]:,.2f}**")
        
        # Add some additional info
        st.info(f"📈 Prediction confidence: This estimate is based on the trained model using all {len(input_data.columns)} features.")
        
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.write("Please check that all inputs are valid.")
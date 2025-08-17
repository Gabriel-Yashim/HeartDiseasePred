import streamlit as st
import pickle
import pandas as pd
import numpy as np


model = pickle.load(open('/Users/Gabriel_Yashim/Documents/GitHub/HeartDiseasePred/GBC_model.pkl', 'rb'))


st.title('Carviovascular Disease Prediction System')
html_temp = """
    <h3 style="color:white;text-align:center;"></h3>
    <div style="background-color:#00246b; padding:10px; margin-bottom:3rem">
        <p style="text-align:justify;">
            Welcome to this simple Carviovascular Disease Prediction System. The system can tell if a person has Carviovascular Disease based on the following features: Age, Glucose, BloodPressure (Systolic and Diastolic), BMI, Diabetes Cholesterol <br> 
            Fill the fields below with the right data to get predictions.
        </p>  
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

def preprocess_input(data):
    # Convert categorical inputs to numerical values
    gender_mapping = {'Female': 1, 'Male': 2}
    cat_mapping = {'Normal': 1, 'Above Normal': 2, 'Well Above Normal': 3}

    data['Gender'] = gender_mapping.get(data['Gender'], 1)
    data['Cholesterol'] = cat_mapping.get(data['Cholesterol'], 1)
    data['Glucose'] = cat_mapping.get(data['Glucose'], 1)

    df = pd.DataFrame(data)
    return df

def main():
    st.title("Alzheimer's Prediction App")

    # Create form for user input
    with st.form(key='input_form'):
        age = st.number_input("Age (Years)", min_value=0,  value=0)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height", min_value=0.0, value=0)
        weight = st.number_input("Weight", min_value=0.0, value=0.0)
        bmi = st.number_input("BMI", min_value=0.0, value=0.0)
        ap_hi = st.number_input("SystolicBP", min_value=0.0, value=0.0)
        ap_lo = st.number_input("DiastolicBP", min_value=0.0, value=0.0)
        cholesterol = st.selectbox("Cholesterol", ['Normal', 'Above Normal', 'Well Above Normal'])
        glucose = st.selectbox("Glucose", ['Normal', 'Above Normal', 'Well Above Normal'])


        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        # Gather input data
        input_data = {
            'Age': age,
            'Gender': gender,
            'Height': height,
            'Weight': weight,
            'BMI': bmi,
            'SystolicBP': ap_hi,
            'DiastolicBP': ap_lo,
            'Cholesterol': cholesterol,
            'Glucose': glucose
        }

        # Preprocess input data
        input_df = preprocess_input(input_data)

        # Make prediction using the XGBoost model
        prediction = model.predict(input_df)

        # Display the result
        if prediction[0] == 1:
            st.write(f"RESULT: You are at risk of Carviovascular Disease")
        else:
            st.write(f"RESULT: You are not at risk of Carviovascular Disease")

if __name__ == '__main__':
    main()

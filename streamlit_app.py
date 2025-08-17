import streamlit as st
import joblib
import pandas as pd


model = joblib.load(open('/Users/Gabriel_Yashim/Documents/GitHub/HeartDiseasePred/GBC_model.joblib', 'rb'))


st.title('Carviovascular Disease Prediction System')
html_temp = """
    <h3 style="color:white;text-align:center;"></h3>
    <div style="background-color:#00246b; padding:50px; margin-bottom:3rem; border-radius:10px;">
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

    data['gender'] = gender_mapping.get(data['gender'], 1)
    data['cholesterol'] = cat_mapping.get(data['cholesterol'], 1)
    data['gluc'] = cat_mapping.get(data['gluc'], 1)

    df = pd.DataFrame([data])
    return df

def main():
    st.title("Carviovascular Disease Prediction App")

    # Create form for user input
    with st.form(key='input_form'):
        age = st.number_input("Age (Years)", min_value=0,  value=0)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height", min_value=0.0, value=0.0)
        weight = st.number_input("Weight", min_value=0.0, value=0.0)
        ap_hi = st.number_input("SystolicBP", min_value=0, value=0)
        ap_lo = st.number_input("DiastolicBP", min_value=0, value=0)
        bmi = st.number_input("BMI", min_value=0.0, value=0.0)
        cholesterol = st.selectbox("Cholesterol", ['Normal', 'Above Normal', 'Well Above Normal'])
        gluc = st.selectbox("Glucose", ['Normal', 'Above Normal', 'Well Above Normal'])

        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        # Gather input data
        input_data = {
            'age': age,
            'gender': gender,
            'height': round(height),
            'weight': weight,
            'bmi': bmi,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'cholesterol': cholesterol,
            'gluc': gluc
        }

        # Preprocess input data
        input_df = preprocess_input(input_data)

        prediction = model.predict(input_df)

        # Display the result
        if prediction[0] == 1:
            st.write(f"RESULT: You are at risk of Carviovascular Disease")
        else:
            st.write(f"RESULT: You are not at risk of Carviovascular Disease")

if __name__ == '__main__':
    main()

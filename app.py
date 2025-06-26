import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="ASD Predictor", page_icon="🧠", layout="centered")

# ===============================
# Loading Trained Model
# ===============================
def load_model():
    model = joblib.load('best_model.joblib')
    feature_columns = joblib.load('feature_columns.joblib')
    scaler = joblib.load('feature_scaler.joblib')
    return model, feature_columns, scaler

# ===============================
# Input Preprocessing
# ===============================
def preprocess_input(input_data, feature_columns, scaler):
    processed_input = {}

    for col in feature_columns:
        if col.startswith('A') and col.endswith('_Score'):
            processed_input[col] = input_data.get(col, 0)
        elif col == 'age':
            processed_input[col] = np.log(float(input_data.get('age', 25.0)) + 1)
        elif col in ['gender', 'jaundice', 'austim', 'used_app_before', 'ethnicity', 'contry_of_res', 'relation', 'ageGroup']:
            processed_input[col] = input_data.get(col, 'unknown')
        elif col in ['result', 'sum_score', 'ind']:
            processed_input[col] = input_data.get(col, 0)
        else:
            processed_input[col] = 0

    full_df = pd.DataFrame([processed_input])[feature_columns]

    for col in full_df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        full_df[col] = le.fit_transform(full_df[col].astype(str))

    try:
        imputer = SimpleImputer(strategy='mean')
        df_imputed = imputer.fit_transform(full_df)
        df_scaled = scaler.transform(df_imputed)
        return df_scaled
    except Exception as e:
        st.error(f"⚠️ Error in preprocessing: {e}")
        st.stop()

# ===============================
# Age Group
# ===============================
def convertAge(age):
    if age < 4:
        return 'Toddler'
    elif age < 12:
        return 'Kid'
    elif age < 18:
        return 'Teenager'
    elif age < 40:
        return 'Young'
    else:
        return 'Senior'

# ===============================
# Streamlit App
# ===============================
def main():
    st.title("ASD Risk Assessment Tool")
    st.markdown("AI-powered assessment for early detection of Autism Spectrum Disorder.")
    st.markdown("---")

    model, feature_columns, scaler = load_model()

    input_data = {}

    with st.expander("📋 Enter Demographic Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            input_data['age'] = st.number_input('Age', min_value=0, max_value=100, value=25)
            input_data['gender'] = st.selectbox('Gender', ['male', 'female'])
            input_data['austim'] = st.selectbox('Diagnosed with Autism', ['yes', 'no'])
        with col2:
            input_data['jaundice'] = st.selectbox('Had Jaundice at Birth', ['yes', 'no'])
            input_data['used_app_before'] = st.selectbox('Used ASD Screening App Before', ['yes', 'no'])
            input_data['result'] = st.number_input('Screening Result Value', -5.0, 100.0, 0.0)

    with st.expander("🧾 Enter Screening Scores (0–10)", expanded=True):
        col1, col2 = st.columns(2)
        for i in range(1, 6):
            input_data[f'A{i}_Score'] = col1.number_input(f'A{i} Score', 0, 10, 0)
        for i in range(6, 11):
            input_data[f'A{i}_Score'] = col2.number_input(f'A{i} Score', 0, 10, 0)

    # Feature engineering
    input_data['sum_score'] = sum(input_data[f'A{i}_Score'] for i in range(1, 11))
    input_data['ind'] = sum([
        input_data['austim'] == 'yes',
        input_data['jaundice'] == 'yes',
        input_data['used_app_before'] == 'yes'
    ])
    input_data['ageGroup'] = convertAge(input_data['age'])

    st.markdown("---")
    if st.button("🔍 Predict ASD"):
        with st.spinner("Processing and predicting..."):
            processed_input = preprocess_input(input_data, feature_columns, scaler)
            prediction = model.predict(processed_input)
            prediction_proba = model.predict_proba(processed_input)[0][1]

        st.subheader("🔎 Prediction Result")
        if prediction[0] == 1:
            st.error("⚠️ High Risk: Potential Autism Spectrum Disorder Detected")
        else:
            st.success("✅ Low Risk: No ASD Detected")

        st.metric(label="Probability of ASD", value=f"{prediction_proba:.2%}")
        st.info("Model Used: Logistic Regression")

# ===============================
# Run
# ===============================
if __name__ == '__main__':
    main()

import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# ## Load the trainer model, scaler pickle, label encoder, and one hot encoder

model = tf.keras.models.load_model('regression_model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    le_gender = pickle.load(f)

with open('oh_encoder_geography.pkl', 'rb') as f:
    oh_geography = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

## streamlit app
st.title('Customer Churn Prediction')

# Input form
geography = st.selectbox('Geography', oh_geography.categories_[0])
gender = st.selectbox('Gender', le_gender.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.slider('Credit Score', 300, 850)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
exited = st.selectbox('Exited', [0, 1])

if st.button('Predict', key='predict_button'):

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [le_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member], 
        'Exited': [exited]
    })

    input_data_geography = oh_geography.transform([[geography]])

    input_data = pd.concat([input_data.reset_index(drop=True), pd.DataFrame(input_data_geography, columns=oh_geography.get_feature_names_out())], axis=1)

    print(input_data)

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    prediction_probability = prediction[0][0]

    st.write(f'Predicted salary of customer: {prediction_probability:.2f}')

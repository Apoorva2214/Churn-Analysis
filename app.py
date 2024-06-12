import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the scaler and model
scaler = joblib.load('scaler.pkl')  # Load the saved scaler
model = joblib.load('churn_predict_model')  # Load your trained model

# Function to make predictions
def predict_churn(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10):
    # Geography encoding
    if p9 == 1:
        Geography_Germany, Geography_Spain = 1, 0
    elif p9 == 2:
        Geography_Germany, Geography_Spain = 0, 1
    elif p9 == 3:
        Geography_Germany, Geography_Spain = 0, 0  # France is the reference category

    features = np.array([[p1, p2, p3, p4, p5, p6, p7, p8, Geography_Germany, Geography_Spain, p10]])
    scaled_features = scaler.transform(features)
    result = model.predict(scaled_features)
    return result[0]

# Streamlit app
st.title("Bank Customer Churn Prediction")

# Create columns for a more compact layout
col1, col2 = st.columns(2)

with col1:
    p1 = st.number_input('Credit Score', min_value=0, max_value=1000, value=600, help="Enter the customer's credit score.")
    p2 = st.number_input('Age', min_value=0, max_value=120, value=35, help="Enter the customer's age.")
    p3 = st.number_input('Tenure', min_value=0, max_value=10, value=5, help="Enter the number of years the customer has been with the bank.")
    p4 = st.number_input('Balance', value=0.0, help="Enter the customer's account balance.")
    p5 = st.number_input('Number of Products', min_value=1, max_value=4, value=1, help="Enter the number of products the customer has.")

with col2:
    p6 = st.selectbox('Has Credit Card', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', help="Does the customer have a credit card?")
    p7 = st.selectbox('Is Active Member', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', help="Is the customer an active member?")
    p8 = st.number_input('Estimated Salary', value=50000.0, help="Enter the customer's estimated salary.")
    p9 = st.selectbox('Geography', options=[1, 2, 3], format_func=lambda x: {1: 'Germany', 2: 'Spain', 3: 'France'}.get(x), help="Select the customer's country.")
    p10 = st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female', help="Select the customer's gender.")

if st.button('Predict'):
    result = predict_churn(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10)
    if result == 1:
        st.success("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")

# Footer for additional information or credits
st.markdown("""
    ---
    ### About
    This app uses a machine learning model to predict whether a bank customer will churn.
    For more information, visit [GitHub](https://github.com/Apoorva2214/Churn-Analysis).
""")

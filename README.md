# Bank Customer Churn Prediction

## Overview
This repository contains a comprehensive project designed to predict and analyze customer churn for a bank. It includes the following components:

### Streamlit App for Churn Prediction:
- **Description**: This app uses a trained machine learning model to predict whether a bank customer is likely to churn based on various input parameters.
- **Input Parameters**:
  - Credit Score
  - Age
  - Tenure
  - Balance
  - Number of Products
  - Has Credit Card
  - Is Active Member
  - Estimated Salary
  - Geography (Germany, Spain, France)
  - Gender (Male, Female)
- **Usage**: Enter the required customer details into the app, and it will output whether the customer is likely to churn.
- **Tech Stack**: Python, Streamlit, Scikit-learn, Joblib for model serialization.

### Power BI Dashboard:
- **Description**: An interactive Power BI dashboard that provides visual insights into customer churn data. It includes various charts and graphs to help understand the factors influencing customer churn.
- **Features**:
  - Churn rate analysis
  - Demographic segmentation
  - Financial metrics visualization
  - Interactive filters and slicers
- **Usage**: Open the Power BI report file to explore the visual analysis of customer churn data.
- **Tech Stack**: Power BI, DAX for data analysis expressions.

### Jupyter Notebook:
- **Description**: A Jupyter notebook (`Churn_Analysis.ipynb`) detailing the data analysis, feature engineering, and model training process.
- **Contents**:
  - Data Cleaning and Preprocessing
  - Exploratory Data Analysis (EDA)
  - Feature Engineering
  - Model Training and Evaluation
- **Tech Stack**: Python, Jupyter, Pandas, Scikit-learn, Matplotlib, Seaborn.

## Files and Directories
- `churn_predict_model.pkl`: The trained machine learning model used for prediction.
- `scaler.pkl`: The scaler used to normalize input data for the model.
- `churn_prediction_app.py`: The Streamlit app for customer churn prediction.
- `Churn Analysis.pbix`: The Power BI report file for visual analysis.
- `Churn_Analysis.ipynb`: The Jupyter notebook for data analysis and model creation.

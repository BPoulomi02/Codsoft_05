
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Title of the app
st.title("Credit Card Fraud Detection with Logistic Regression")

# File uploader for the user to upload their dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    
    # Show a preview of the dataset
    st.write("Dataset Preview:", data.head())
    
    # Check for missing values and handle them
    if data.isnull().sum().any():
        st.write("Handling missing values...")
        data = data.dropna()
    
    # Standardize 'Amount' and 'Time' columns
    st.write("Standardizing 'Amount' and 'Time' columns...")
    scaler = StandardScaler()
    data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data['Time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))
    
    # Separate features and target variable
    X = data.drop(columns=['Class'])
    y = data['Class']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Handle class imbalance by downsampling the majority class
    train_data = pd.concat([X_train, y_train], axis=1)
    not_fraud = train_data[train_data.Class == 0]
    fraud = train_data[train_data.Class == 1]
    
    st.write("Downsampling the majority class to handle class imbalance...")
    not_fraud_downsampled = resample(not_fraud,
                                     replace=False,
                                     n_samples=len(fraud),
                                     random_state=42)
    
    # Combine the downsampled not_fraud with fraud to create a balanced dataset
    downsampled_data = pd.concat([not_fraud_downsampled, fraud])
    
    # Separate the downsampled dataset into features and target variable
    X_train_balanced = downsampled_data.drop(columns=['Class'])
    y_train_balanced = downsampled_data['Class']
    
    # Train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train_balanced, y_train_balanced)
    
    # Predict the test set
    y_pred = model.predict(X_test)
    
    # Display the classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("Classification Report:")
    st.json(report)
    
    st.success("Model training and evaluation completed successfully!")

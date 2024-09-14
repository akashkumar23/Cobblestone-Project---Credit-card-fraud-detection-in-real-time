import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Function to load the data with error handling
@st.cache(allow_output_mutation=True)
def load_data(file):
    try:
        data = pd.read_csv(file)
        return data
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Function to train the model with error handling
def train_model(data):
    try:
        # Check if "Class" column exists
        if 'Class' not in data.columns:
            st.error("The dataset does not contain the required 'Class' column.")
            return None, None, None

        # separate legitimate and fraudulent transactions
        legit = data[data.Class == 0]
        fraud = data[data.Class == 1]

        # check if there are any fraudulent transactions
        if len(fraud) == 0 or len(legit) == 0:
            st.error("The dataset does not contain sufficient legitimate or fraudulent transactions.")
            return None, None, None

        # undersample legitimate transactions to balance the classes
        legit_sample = legit.sample(n=len(fraud), random_state=2)
        data_balanced = pd.concat([legit_sample, fraud], axis=0)

        # split data into training and testing sets
        X = data_balanced.drop(columns="Class", axis=1)
        y = data_balanced["Class"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=2
        )

        # train logistic regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # evaluate model performance
        train_acc = accuracy_score(model.predict(X_train), y_train)
        test_acc = accuracy_score(model.predict(X_test), y_test)

        return model, train_acc, test_acc
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None

# Function to parse transaction string with error handling
def parse_transaction_string(transaction_string, feature_names):
    try:
        values = transaction_string.split(",")
        if len(values) != len(feature_names):
            st.error("The number of features entered does not match the number of features in the dataset.")
            return None

        transaction = {}
        for i in range(len(values)):
            try:
                transaction[feature_names[i]] = float(values[i])
            except ValueError:
                st.error(f"Invalid input: '{values[i]}' is not a valid number.")
                return None

        return transaction
    except Exception as e:
        st.error(f"Error parsing transaction: {e}")
        return None

# Main Streamlit app
st.title("Credit Card Fraud Detection")

# File uploader for CSV file
file = st.file_uploader("Upload a CSV file containing credit card transaction data:")
if file is not None:
    # Load data
    data = load_data(file)
    
    if data is not None:
        st.write("Data shape:", data.shape)
        
        # Train the model if data is loaded successfully
        model, train_acc, test_acc = train_model(data)
        
        if model is not None:
            st.write("Training accuracy:", train_acc)
            st.write("Test accuracy:", test_acc)

            # Allow user to input transaction features for prediction
            st.subheader("Check a transaction")
            feature_names = data.drop(columns="Class", axis=1).columns

            transaction_string = st.text_input(
                "Enter transaction features (comma-separated)",
                "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"
            )

            # Parse the user input
            transaction = parse_transaction_string(transaction_string, feature_names)

            if transaction is not None:
                transaction_df = pd.DataFrame([transaction])

                # Make a prediction using the trained model
                prediction = model.predict(transaction_df)
                prediction_label = "Fraudulent" if prediction[0] == 1 else "Legitimate"
                st.write(f"The transaction is predicted to be: **{prediction_label}**")
else:
    st.warning("Please upload a CSV file to proceed.")

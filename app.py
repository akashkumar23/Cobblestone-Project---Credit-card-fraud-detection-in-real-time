import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Function to load data with error handling
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('creditcard.csv')
        return data
    except FileNotFoundError:
        st.error("Error: The 'creditcard.csv' file was not found. Please ensure the file is in the correct location.")
        return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Function to train the model with error handling
def train_model(data):
    try:
        legit = data[data.Class == 0]
        fraud = data[data.Class == 1]

        # Check if data is imbalanced or missing
        if len(legit) == 0 or len(fraud) == 0:
            st.error("Error: Dataset must contain both legitimate and fraudulent transactions.")
            return None, None, None

        # Undersample legitimate transactions to balance classes
        legit_sample = legit.sample(n=len(fraud), random_state=2)
        balanced_data = pd.concat([legit_sample, fraud], axis=0)

        # Split data into features (X) and labels (y)
        X = balanced_data.drop(columns="Class", axis=1)
        y = balanced_data["Class"]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

        # Train logistic regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Evaluate model performance
        train_acc = accuracy_score(model.predict(X_train), y_train)
        test_acc = accuracy_score(model.predict(X_test), y_test)

        return model, train_acc, test_acc
    except Exception as e:
        st.error(f"Error during model training: {e}")
        return None, None, None

# Main Streamlit App
st.title("Credit Card Fraud Detection Model")

# Load data
data = load_data()

if data is not None:
    # Train model if data is successfully loaded
    model, train_acc, test_acc = train_model(data)

    if model is not None:
        # Display model accuracy
        st.write(f"Training Accuracy: {train_acc:.2f}")
        st.write(f"Test Accuracy: {test_acc:.2f}")

        # Create input field for user to enter feature values
        st.write("Enter the following features (comma-separated) to check if the transaction is legitimate or fraudulent:")

        input_df = st.text_input('Input all features (comma-separated)')

        submit = st.button("Submit")

        if submit:
            try:
                # Process input and make sure it has the correct number of features
                input_df_lst = input_df.split(',')
                if len(input_df_lst) != X.shape[1]:  # Ensure correct number of features
                    st.error(f"Error: You must input exactly {X.shape[1]} features. You entered {len(input_df_lst)}.")
                else:
                    # Convert input values to floats
                    features = np.array(input_df_lst, dtype=np.float64)
                    
                    # Make prediction
                    prediction = model.predict(features.reshape(1, -1))

                    # Display result
                    if prediction[0] == 0:
                        st.success("Legitimate transaction")
                    else:
                        st.warning("Fraudulent transaction")
            except ValueError:
                st.error("Error: Please ensure all feature values are valid numbers.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
else:
    st.warning("Please upload a valid 'creditcard.csv' file to proceed.")

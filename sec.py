'''import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to load data from a CSV file
def load_data(file):
    return pd.read_csv(file)

# Function to train models on the loaded data
def train_models(data):
    # Separate legitimate and fraudulent transactions
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]

    # Undersample legitimate transactions to balance the classes
    legit_sample = legit.sample(n=len(fraud), random_state=2)
    data = pd.concat([legit_sample, fraud], axis=0)

    # Split data into features and labels
    X = data.drop(columns="Class", axis=1)
    y = data["Class"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

    # Train multiple models
    logreg_model = LogisticRegression()
    logreg_model.fit(X_train, y_train)
    train_acc_logreg = accuracy_score(logreg_model.predict(X_train), y_train)
    test_acc_logreg = accuracy_score(logreg_model.predict(X_test), y_test)

    dt_model = DecisionTreeClassifier(random_state=2)
    dt_model.fit(X_train, y_train)
    train_acc_dt = accuracy_score(dt_model.predict(X_train), y_train)
    test_acc_dt = accuracy_score(dt_model.predict(X_test), y_test)

    rf_model = RandomForestClassifier(random_state=2)
    rf_model.fit(X_train, y_train)
    train_acc_rf = accuracy_score(rf_model.predict(X_train), y_train)
    test_acc_rf = accuracy_score(rf_model.predict(X_test), y_test)

    return logreg_model, dt_model, rf_model, X, train_acc_logreg, train_acc_dt, train_acc_rf, test_acc_dt, test_acc_logreg, test_acc_rf



# Streamlit app title
st.title("Credit Card Fraud Detection")

# File uploader to upload dataset
file = st.file_uploader("Upload a CSV file containing credit card transaction data:")

if file:
    # Load the data
    data = load_data(file)

    # Train models on the uploaded dataset
    logreg_model, dt_model, rf_model, X = train_models(data)

    st.success("Dataset loaded and models trained successfully!")

    # Dropdown to select the model
    model_choice = st.selectbox("Choose the model for prediction:", 
                                ["Logistic Regression", "Decision Tree", "Random Forest"])

    # Input field for comma-separated feature values
    input_features = st.text_input(f"Enter {X.shape[1]} comma-separated feature values:")

    # Button for making predictions
    if st.button("Make Prediction"):
        try:
            # Parse input feature values
            features = np.array([float(x) for x in input_features.split(',')], dtype=np.float64)
            
            if len(features) != X.shape[1]:
                st.error(f"Please enter exactly {X.shape[1]} feature values.")
            else:
                # Reshape the input features
                input_data = features.reshape(1, -1)

                # Select the model for prediction
                if model_choice == "Logistic Regression":
                    prediction = logreg_model.predict(input_data)
                    
                
                elif model_choice == "Decision Tree":
                    prediction = dt_model.predict(input_data)
                    
                
                elif model_choice == "Random Forest":
                    prediction = rf_model.predict(input_data)
                    

                # Display prediction result
                if prediction[0] == 0:
                    st.success("Prediction: Legitimate Transaction")
                else:
                    st.error("Prediction: Fraudulent Transaction")
        except ValueError:
            st.error("Please enter valid numeric values separated by commas.")
else:
    st.warning("Please upload a CSV file to proceed.")
'''
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to load data from a CSV file
def load_data(file):
    return pd.read_csv(file)

# Function to train models on the loaded data and return accuracy scores
def train_models(data):
    # Separate legitimate and fraudulent transactions
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]

    # Undersample legitimate transactions to balance the classes
    legit_sample = legit.sample(n=len(fraud), random_state=2)
    data = pd.concat([legit_sample, fraud], axis=0)

    # Split data into features and labels
    X = data.drop(columns="Class", axis=1)
    y = data["Class"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

    # Train Logistic Regression model
    logreg_model = LogisticRegression()
    logreg_model.fit(X_train, y_train)
    logreg_train_acc = accuracy_score(y_train, logreg_model.predict(X_train))
    logreg_test_acc = accuracy_score(y_test, logreg_model.predict(X_test))

    # Train Decision Tree model
    dt_model = DecisionTreeClassifier(random_state=2)
    dt_model.fit(X_train, y_train)
    dt_train_acc = accuracy_score(y_train, dt_model.predict(X_train))
    dt_test_acc = accuracy_score(y_test, dt_model.predict(X_test))

    # Train Random Forest model
    rf_model = RandomForestClassifier(random_state=2)
    rf_model.fit(X_train, y_train)
    rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train))
    rf_test_acc = accuracy_score(y_test, rf_model.predict(X_test))

    return (logreg_model, dt_model, rf_model, X, 
            logreg_train_acc, logreg_test_acc, 
            dt_train_acc, dt_test_acc, 
            rf_train_acc, rf_test_acc)

# Streamlit app title
st.title("Credit Card Fraud Detection")

# File uploader to upload dataset
file = st.file_uploader("Upload a CSV file containing credit card transaction data:")

if file:
    # Load the data
    data = load_data(file)

    # Train models on the uploaded dataset and get accuracies
    (logreg_model, dt_model, rf_model, X, 
     logreg_train_acc, logreg_test_acc, 
     dt_train_acc, dt_test_acc, 
     rf_train_acc, rf_test_acc) = train_models(data)

    st.success("Dataset loaded and models trained successfully!")

    # Display accuracy scores
    st.subheader("Model Accuracy Scores")
    st.write(f"**Logistic Regression** - Training Accuracy: {logreg_train_acc*100:.4f}%, Testing Accuracy: {logreg_test_acc*100:.4f}%")
    st.write(f"**Decision Tree** - Training Accuracy: {dt_train_acc*100:.4f}%, Testing Accuracy: {dt_test_acc*100:.4f}%")
    st.write(f"**Random Forest** - Training Accuracy: {rf_train_acc*100:.4f}%, Testing Accuracy: {rf_test_acc*100:.4f}%")

    # Dropdown to select the model
    model_choice = st.selectbox("Choose the model for prediction:", 
                                ["Logistic Regression", "Decision Tree", "Random Forest"])

    # Input field for comma-separated feature values
    input_features = st.text_input(f"Enter {X.shape[1]} comma-separated feature values:")

    # Button for making predictions
    if st.button("Make Prediction"):
        try:
            # Parse input feature values
            features = np.array([float(x) for x in input_features.split(',')], dtype=np.float64)
            
            if len(features) != X.shape[1]:
                st.error(f"Please enter exactly {X.shape[1]} feature values.")
            else:
                # Reshape the input features
                input_data = features.reshape(1, -1)

                # Select the model for prediction
                if model_choice == "Logistic Regression":
                    prediction = logreg_model.predict(input_data)
                elif model_choice == "Decision Tree":
                    prediction = dt_model.predict(input_data)
                elif model_choice == "Random Forest":
                    prediction = rf_model.predict(input_data)

                # Display prediction result
                if prediction[0] == 0:
                    st.success("Prediction: Legitimate Transaction")
                else:
                    st.error("Prediction: Fraudulent Transaction")
        except ValueError:
            st.error("Please enter valid numeric values separated by commas.")
else:
    st.warning("Please upload a CSV file to proceed.")

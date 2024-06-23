import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Function to load and preprocess data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)

    # Select relevant features and target
    features = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
    target = 'is_fraud'

    X = data[features]
    y = data[target]

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ensure arrays are writeable
    X_scaled = np.copy(X_scaled)
    y = np.copy(y)

    return X_scaled, y, data

# Function to train models and evaluate them
def train_models(X_train, y_train, X_test, y_test):
    results = {}

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_log_reg = log_reg.predict(X_test)
    results['Logistic Regression'] = {
        'Accuracy': accuracy_score(y_test, y_pred_log_reg),
        'Precision': precision_score(y_test, y_pred_log_reg),
        'Recall': recall_score(y_test, y_pred_log_reg),
        'F1 Score': f1_score(y_test, y_pred_log_reg),
        'ROC AUC': roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])
    }

    # Decision Tree
    dec_tree = DecisionTreeClassifier(random_state=42)
    dec_tree.fit(X_train, y_train)
    y_pred_dec_tree = dec_tree.predict(X_test)
    results['Decision Tree'] = {
        'Accuracy': accuracy_score(y_test, y_pred_dec_tree),
        'Precision': precision_score(y_test, y_pred_dec_tree),
        'Recall': recall_score(y_test, y_pred_dec_tree),
        'F1 Score': f1_score(y_test, y_pred_dec_tree),
        'ROC AUC': roc_auc_score(y_test, dec_tree.predict_proba(X_test)[:, 1])
    }

    # Random Forest
    rand_forest = RandomForestClassifier(random_state=42)
    rand_forest.fit(X_train, y_train)
    y_pred_rand_forest = rand_forest.predict(X_test)
    results['Random Forest'] = {
        'Accuracy': accuracy_score(y_test, y_pred_rand_forest),
        'Precision': precision_score(y_test, y_pred_rand_forest),
        'Recall': recall_score(y_test, y_pred_rand_forest),
        'F1 Score': f1_score(y_test, y_pred_rand_forest),
        'ROC AUC': roc_auc_score(y_test, rand_forest.predict_proba(X_test)[:, 1])
    }

    return results

# Main Streamlit app
def main():
    st.title("Credit Card Fraud Detection")

    # Upload dataset
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        X, y, data = load_data(uploaded_file)
        
        # Display dataset
        st.subheader("Credit Card Transaction Data")
        st.write(data.head())

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train models and get results
        results = train_models(X_train, y_train, X_test, y_test)

        # Display results
        st.subheader("Model Evaluation Metrics")
        metrics_df = pd.DataFrame(results).T
        st.write(metrics_df)

        # Allow users to input transaction data for prediction within a form
        st.subheader("Predict Fraudulent Transaction")
        with st.form(key='transaction_form'):
            st.write("Enter transaction details:")
            amt = st.number_input('Amount', value=0.0)
            lat = st.number_input('Latitude', value=0.0)
            long = st.number_input('Longitude', value=0.0)
            city_pop = st.number_input('City Population', value=0)
            unix_time = st.number_input('Unix Time', value=0)
            merch_lat = st.number_input('Merchant Latitude', value=0.0)
            merch_long = st.number_input('Merchant Longitude', value=0.0)
            
            model_choice = st.selectbox("Choose model", ["Logistic Regression", "Decision Tree", "Random Forest"])
            
            submit_button = st.form_submit_button(label='Submit')
        
        if submit_button:
            transaction_data = np.array([amt, lat, long, city_pop, unix_time, merch_lat, merch_long]).reshape(1, -1)
            
            # Standardize the transaction data using the same scaler
            scaler = StandardScaler()
            scaler.fit(X)  # Fit on the entire dataset
            transaction_data_scaled = scaler.transform(transaction_data)
            
            # Select model for prediction
            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_choice == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42)
            else:
                model = RandomForestClassifier(random_state=42)
            
            # Train the selected model
            model.fit(X_train, y_train)
            prediction = model.predict(transaction_data_scaled)
            prediction_proba = model.predict_proba(transaction_data_scaled)[:, 1]
            
            # Display prediction result
            st.subheader("Prediction")
            if prediction[0] == 1:
                st.error("This transaction is predicted to be FRAUDULENT.")
            else:
                st.success("This transaction is predicted to be LEGITIMATE.")
            
            st.write(f"Prediction probability: {prediction_proba[0]:.2f}")

if __name__ == '__main__':
    main()

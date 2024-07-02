# Credit Card Fraud Detection
This project aims to develop a machine learning model to detect fraudulent credit card transactions. Using a dataset containing information about credit card transactions, we apply algorithms such as Logistic Regression, Decision Trees, and Random Forests to classify transactions as fraudulent or legitimate.

## Table of Contents
 - Project Overview
 - Features
 - Installation
 - Modeling

Credit card fraud detection is crucial for financial institutions to prevent unauthorized transactions and reduce financial losses. By analyzing transaction data, we can identify patterns that indicate fraudulent activity. This project uses transaction details to train predictive models.

### Features
Data Preprocessing: Handling missing values, feature engineering, and normalization.
Exploratory Data Analysis (EDA): Visualizing data distributions and relationships.
Model Training: Implementing Logistic Regression, Decision Trees, and Random Forests.
Model Evaluation: Assessing model performance using metrics like accuracy, precision, recall, and F1-score.
Streamlit Web App: Interactive interface to visualize data and detect fraudulent transactions.

### Create a virtual environment:

    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    
### Install the required packages:

    pip install -r requirements.txt

### Run the Streamlit application:

    streamlit run app.py

## Modeling
The project includes the following steps:

- Data Preprocessing: Clean and preprocess the data to handle missing values, encode categorical features, and normalize numerical features.
- Exploratory Data Analysis (EDA): Gain insights into data distributions and relationships using visualizations.
- Model Training: Train models using Logistic Regression, Decision Trees, and Random Forests.
- Model Evaluation: Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.

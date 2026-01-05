"""
Data Preprocessing Module
Handles data loading, cleaning, encoding, and scaling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def load_and_preprocess_data(file_path, test_size=0.2, random_state=42):
    """
    Load and preprocess customer churn data
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    test_size : float
        Proportion of test set
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    X_train, X_test, y_train, y_test, scaler
    """
    
    # Load data
    print(f"  → Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Display basic info
    print(f"  → Dataset shape: {df.shape}")
    print(f"  → Columns: {list(df.columns)}")
    
    # Handle missing values
    print("  → Handling missing values...")
    df = df.dropna()
    
    # Identify target column (common names)
    target_col = None
    for col in ['Churn', 'churn', 'Exited', 'exited', 'Attrition_Flag']:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        raise ValueError("Target column not found. Expected 'Churn' or similar.")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode target variable if needed
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Remove ID columns if present
    id_cols = ['CustomerID', 'customerID', 'customer_id', 'id', 'ID']
    for col in id_cols:
        if col in X.columns:
            X = X.drop(columns=[col])
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    print(f"  → Encoding {len(categorical_cols)} categorical columns...")
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Split the data
    print("  → Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    print("  → Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train, X_test, y_train, y_test, scaler

def load_scaler():
    """Load saved scaler"""
    return joblib.load('models/scaler.pkl')
"""
Prediction Module
Makes churn predictions for new customers
"""

import numpy as np
import pandas as pd
from src.model_training import load_model
from src.data_preprocessing import load_scaler

def predict_churn(model, X_new):
    """
    Predict churn for new customer data
    
    Parameters:
    -----------
    model : trained model
        Model to use for prediction
    X_new : array
        New customer data (already scaled and selected features)
    
    Returns:
    --------
    predictions : array
        Churn predictions (0 = No Churn, 1 = Churn)
    """
    
    predictions = model.predict(X_new)
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_new)
        return predictions, probabilities
    
    return predictions

def predict_single_customer(customer_data):
    """
    Predict churn for a single customer
    
    Parameters:
    -----------
    customer_data : dict or array
        Customer features
    
    Returns:
    --------
    prediction : int
        0 (No Churn) or 1 (Churn)
    probability : float
        Churn probability
    """
    
    # Load saved model and scaler
    model = load_model('churn_model.pkl')
    scaler = load_scaler()
    
    # Convert to array if dict
    if isinstance(customer_data, dict):
        customer_data = np.array(list(customer_data.values())).reshape(1, -1)
    
    # Scale data
    customer_scaled = scaler.transform(customer_data)
    
    # Predict
    prediction = model.predict(customer_scaled)[0]
    
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(customer_scaled)[0][1]
        return prediction, probability
    
    return prediction, None

def interpret_prediction(prediction, probability=None):
    """
    Interpret prediction results
    
    Parameters:
    -----------
    prediction : int
        0 or 1
    probability : float
        Churn probability
    
    Returns:
    --------
    interpretation : str
    """
    
    if prediction == 1:
        risk_level = "HIGH RISK" if probability and probability > 0.7 else "MODERATE RISK"
        message = f"⚠️  Customer is likely to CHURN ({risk_level})"
        if probability:
            message += f"\n   Churn Probability: {probability*100:.2f}%"
    else:
        message = f"✓  Customer is likely to STAY"
        if probability:
            message += f"\n   Retention Probability: {(1-probability)*100:.2f}%"
    
    return message

# Example usage
if __name__ == "__main__":
    # Example: Predict for sample data
    sample_data = np.random.randn(1, 10)  # Replace with actual features
    
    try:
        model = load_model('churn_model.pkl')
        prediction = predict_churn(model, sample_data)
        print(interpret_prediction(prediction[0]))
    except:
        print("Model not found. Please run main.py first.")
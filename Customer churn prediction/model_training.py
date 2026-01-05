"""
Model Training Module
Trains multiple ML models for churn prediction
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
import os

def train_models(X_train, y_train):
    """
    Train multiple classification models
    
    Parameters:
    -----------
    X_train : array
        Training features
    y_train : array
        Training labels
    
    Returns:
    --------
    models : dict
        Dictionary of trained models
    """
    
    models = {}
    
    # 1. Logistic Regression
    print("  → Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    
    # 2. Random Forest
    print("  → Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    # 3. Gradient Boosting
    print("  → Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    
    # 4. Decision Tree
    print("  → Training Decision Tree...")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    models['Decision Tree'] = dt
    
    # 5. Support Vector Machine
    print("  → Training SVM...")
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    models['SVM'] = svm
    
    # 6. Naive Bayes
    print("  → Training Naive Bayes...")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    models['Naive Bayes'] = nb
    
    return models

def save_model(model, filename='churn_model.pkl'):
    """Save trained model"""
    os.makedirs('models', exist_ok=True)
    filepath = os.path.join('models', filename)
    joblib.dump(model, filepath)
    print(f"  → Model saved to {filepath}")

def load_model(filename='churn_model.pkl'):
    """Load saved model"""
    filepath = os.path.join('models', filename)
    return joblib.load(filepath)
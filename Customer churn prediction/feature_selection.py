"""
Feature Selection Module
Identifies and selects important features for churn prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import os

def select_features(X_train, y_train, X_test, k=10, method='random_forest'):
    """
    Select top k important features
    
    Parameters:
    -----------
    X_train : array
        Training features
    y_train : array
        Training labels
    X_test : array
        Test features
    k : int
        Number of features to select
    method : str
        Feature selection method ('random_forest', 'chi2', 'mutual_info')
    
    Returns:
    --------
    X_train_selected, X_test_selected, selected_feature_indices
    """
    
    print(f"  → Using {method} for feature selection...")
    
    if method == 'random_forest':
        # Random Forest Feature Importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_
        
        # Select top k features
        indices = np.argsort(importances)[::-1][:k]
        
    elif method == 'chi2':
        # Chi-square test (works with non-negative features)
        X_train_positive = X_train - X_train.min() + 1e-10
        selector = SelectKBest(chi2, k=k)
        selector.fit(X_train_positive, y_train)
        indices = selector.get_support(indices=True)
        
    elif method == 'mutual_info':
        # Mutual Information
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        indices = np.argsort(mi_scores)[::-1][:k]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply feature selection
    X_train_selected = X_train[:, indices]
    X_test_selected = X_test[:, indices]
    
    # Visualize feature importance
    plot_feature_importance(indices, importances if method == 'random_forest' else None)
    
    print(f"  → Selected feature indices: {indices}")
    
    return X_train_selected, X_test_selected, indices

def plot_feature_importance(indices, importances=None):
    """Plot and save feature importance"""
    os.makedirs('outputs', exist_ok=True)
    
    if importances is not None:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(indices)), importances[indices])
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.title('Top Feature Importances')
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png', dpi=300)
        plt.close()
        print("  → Feature importance plot saved to outputs/feature_importance.png")
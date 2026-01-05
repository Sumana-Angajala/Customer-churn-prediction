"""
Model Evaluation Module
Evaluates models and generates performance metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_curve, auc, roc_auc_score)
import os
from src.model_training import save_model

def evaluate_model(models, X_test, y_test):
    """
    Evaluate all models and return the best one
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : array
        Test features
    y_test : array
        True labels
    
    Returns:
    --------
    best_model, best_model_name
    """
    
    os.makedirs('outputs', exist_ok=True)
    
    results = []
    best_accuracy = 0
    best_model = None
    best_model_name = None
    
    print("\n" + "="*70)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*70)
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        print(f"\n{name}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
    
    print("\n" + "="*70)
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/model_comparison.csv', index=False)
    print("\n✓ Results saved to outputs/model_comparison.csv")
    
    # Generate detailed report for best model
    generate_detailed_report(best_model, best_model_name, X_test, y_test)
    
    # Save best model
    save_model(best_model, 'churn_model.pkl')
    
    return best_model, best_model_name

def generate_detailed_report(model, model_name, X_test, y_test):
    """Generate detailed evaluation report for best model"""
    
    y_pred = model.predict(X_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=300)
    plt.close()
    
    # Classification Report
    report = classification_report(y_test, y_pred)
    
    # Save to text file
    with open('outputs/accuracy_report.txt', 'w') as f:
        f.write(f"BEST MODEL: {model_name}\n")
        f.write("="*60 + "\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write("-"*60 + "\n")
        f.write(report)
        f.write("\n\nCONFUSION MATRIX:\n")
        f.write("-"*60 + "\n")
        f.write(str(cm))
    
    print("✓ Confusion matrix saved to outputs/confusion_matrix.png")
    print("✓ Detailed report saved to outputs/accuracy_report.txt")

def plot_roc_curve(model, X_test, y_test, model_name):
    """Plot ROC curve"""
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig('outputs/roc_curve.png', dpi=300)
        plt.close()
"""
Customer Churn Prediction - Main Pipeline
Execute the complete ML workflow
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from src.data_preprocessing import load_and_preprocess_data
from src.feature_selection import select_features
from src.model_training import train_models
from src.model_evaluation import evaluate_model
from src.predict import predict_churn

def main():
    print("=" * 60)
    print("🎯 CUSTOMER CHURN PREDICTION SYSTEM")
    print("=" * 60)
    
    # Step 1: Data Loading & Preprocessing
    print("\n[1/5] Loading and Preprocessing Data...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('data/customer_churn.csv')
    print(f"✓ Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # Step 2: Feature Selection
    print("\n[2/5] Selecting Important Features...")
    X_train_selected, X_test_selected, selected_features = select_features(X_train, y_train, X_test)
    print(f"✓ Selected {len(selected_features)} important features")
    
    # Step 3: Model Training
    print("\n[3/5] Training Multiple Models...")
    models = train_models(X_train_selected, y_train)
    print(f"✓ Trained {len(models)} models successfully")
    
    # Step 4: Model Evaluation
    print("\n[4/5] Evaluating Models...")
    best_model, best_model_name = evaluate_model(models, X_test_selected, y_test)
    print(f"✓ Best Model: {best_model_name}")
    
    # Step 5: Make Predictions
    print("\n[5/5] Making Predictions...")
    sample_data = X_test_selected[:5]
    predictions = predict_churn(best_model, sample_data)
    print(f"✓ Sample predictions: {predictions}")
    
    print("\n" + "=" * 60)
    print("✅ PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n📊 Check 'outputs/' folder for results")
    print("💾 Model saved in 'models/' folder")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        sys.exit(1)
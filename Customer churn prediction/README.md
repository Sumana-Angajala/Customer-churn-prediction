# 📉 Customer Churn Prediction

## 🎯 Project Overview
A Machine Learning project to predict customer churn using classification algorithms. This system analyzes historical customer data to identify patterns and predict which customers are likely to leave a service.

## 🌟 Key Features
- ✅ Multiple ML algorithms (Logistic Regression, Random Forest, SVM, etc.)
- ✅ Feature selection and importance analysis
- ✅ Comprehensive model evaluation with metrics
- ✅ Visualization of results (confusion matrix, ROC curves)
- ✅ Modular and clean code structure
- ✅ Easy to understand and modify

## 📂 Project Structure
```
Customer-Churn-Prediction/
│
├── data/                          # Dataset folder
│   └── customer_churn.csv
│
├── src/                           # Source code modules
│   ├── data_preprocessing.py      # Data cleaning & encoding
│   ├── feature_selection.py       # Feature importance
│   ├── model_training.py          # Train ML models
│   ├── model_evaluation.py        # Performance metrics
│   └── predict.py                 # Make predictions
│
├── models/                        # Saved models
│   ├── churn_model.pkl
│   └── scaler.pkl
│
├── outputs/                       # Results & visualizations
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── accuracy_report.txt
│   └── model_comparison.csv
│
├── notebooks/                     # Jupyter notebooks
│   └── churn_analysis.ipynb
│
├── main.py                        # Main execution file
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset
Place your `customer_churn.csv` file in the `data/` folder.

**Dataset Requirements:**
- Must contain a target column named: `Churn`, `churn`, `Exited`, or `Attrition_Flag`
- Features can be numerical or categorical
- Missing values will be handled automatically

### 3. Run the Project
```bash
python main.py
```

### 4. View Results
- Check `outputs/` folder for visualizations and reports
- Model saved in `models/churn_model.pkl`

## 📊 Dataset Information

### Sample Dataset Columns
- **CustomerID**: Unique identifier
- **Gender**: Male/Female
- **Age**: Customer age
- **Tenure**: Months with company
- **Balance**: Account balance
- **NumOfProducts**: Number of products purchased
- **HasCrCard**: Credit card status (0/1)
- **IsActiveMember**: Active status (0/1)
- **EstimatedSalary**: Annual salary
- **Churn**: Target variable (0 = Stayed, 1 = Left)

### Where to Get Dataset?
1. **Kaggle**: Search for "Customer Churn Dataset"
2. **UCI Machine Learning Repository**
3. **Sample Dataset**: Use Telco Customer Churn dataset

## 🧠 Machine Learning Models Used

1. **Logistic Regression** - Simple and interpretable
2. **Random Forest** - Ensemble method with high accuracy
3. **Gradient Boosting** - Advanced boosting technique
4. **Decision Tree** - Easy to visualize
5. **Support Vector Machine (SVM)** - Effective for complex boundaries
6. **Naive Bayes** - Probabilistic classifier

## 📈 Performance Metrics

The system evaluates models using:
- ✅ **Accuracy**: Overall correctness
- ✅ **Precision**: Positive prediction accuracy
- ✅ **Recall**: Ability to find all churned customers
- ✅ **F1-Score**: Balance between precision and recall
- ✅ **Confusion Matrix**: Detailed performance breakdown
- ✅ **ROC Curve**: Trade-off visualization

## 📝 Example Usage

### Predict for New Customer
```python
from src.predict import predict_single_customer, interpret_prediction

# Customer data
customer = {
    'Age': 35,
    'Tenure': 5,
    'Balance': 50000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 80000
}

prediction, probability = predict_single_customer(customer)
print(interpret_prediction(prediction, probability))
```

## 🎯 Results (Example)

```
MODEL PERFORMANCE COMPARISON
======================================
Random Forest:
  Accuracy:  0.8650
  Precision: 0.8621
  Recall:    0.8650
  F1-Score:  0.8630

Best Model: Random Forest
```

## 🚀 Future Enhancements
- [ ] Deep Learning models (LSTM, Neural Networks)
- [ ] Real-time prediction API
- [ ] Web dashboard for visualization
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] SHAP values for model explainability

## 📚 References
- Scikit-learn Documentation
- Machine Learning Mastery
- Kaggle Churn Prediction Competitions

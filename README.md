<div align="center">

# Loan Approval Predictor

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pytest](https://img.shields.io/badge/Tests-75_passed-brightgreen?style=flat&logo=pytest)
![Coverage](https://img.shields.io/badge/Coverage-100%25-brightgreen?style=flat)

**An end-to-end machine learning pipeline that predicts loan application approval status using applicant financial and demographic data.**

</div>

---

## Overview

This project builds a predictive model to determine whether a loan application will be approved or rejected based on features like applicant income, credit history, loan amount, employment status, and more.

The pipeline covers the full ML lifecycle:
- **Data Preprocessing** -- missing value imputation, categorical encoding
- **Feature Engineering** -- outlier removal (z-score), skewness correction (Yeo-Johnson), feature scaling
- **Model Training** -- Logistic Regression and Decision Tree classifiers
- **Hyperparameter Tuning** -- GridSearchCV for optimal model performance
- **Evaluation** -- accuracy, confusion matrix, classification reports
- **Unit Testing** -- 75 tests with 100% code coverage

## Dataset

The dataset contains **614 loan applications** with 12 features and 1 target variable:

| Feature | Type | Description |
|---------|------|-------------|
| Gender | Categorical | Male / Female |
| Married | Categorical | Applicant marital status |
| Dependents | Categorical | Number of dependents (0, 1, 2, 3+) |
| Education | Categorical | Graduate / Not Graduate |
| Self_Employed | Categorical | Yes / No |
| ApplicantIncome | Numerical | Applicant's income |
| CoapplicantIncome | Numerical | Co-applicant's income |
| LoanAmount | Numerical | Loan amount (in thousands) |
| Loan_Amount_Term | Numerical | Term of loan (in months) |
| Credit_History | Categorical | Credit history (1 = good, 0 = bad) |
| Property_Area | Categorical | Urban / Semiurban / Rural |
| **Loan_Status** | **Target** | **Y (Approved) / N (Rejected)** |

## Project Structure

```
Loan-Approval-Predictor-/
├── src/
│   ├── data_preprocessing.py    # Data loading, imputation, encoding
│   ├── feature_engineering.py   # Outlier removal, scaling, transforms
│   └── model.py                 # Training, evaluation, tuning
├── tests/
│   ├── conftest.py              # Shared fixtures and sample data
│   ├── test_data_preprocessing.py
│   ├── test_feature_engineering.py
│   └── test_model.py
├── Evaluation Project 6 Loan Application Status Prediction-Copy1.ipynb
├── pyproject.toml
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Ritesh200508/Loan-Approval-Predictor-.git
cd Loan-Approval-Predictor-

# Install dependencies
pip install pandas numpy scikit-learn scipy

# Install dev dependencies (for testing)
pip install pytest pytest-cov
```

## Usage

### Using the Python modules

```python
from src.data_preprocessing import load_data, preprocess_pipeline
from src.feature_engineering import feature_engineering_pipeline
from src.model import train_logistic_regression, evaluate_model
from sklearn.model_selection import train_test_split

# Load and preprocess
df = load_data("loan_prediction.xls")
df, encoders = preprocess_pipeline(df)

# Feature engineering
X_scaled, y, scaler, transformer, _ = feature_engineering_pipeline(df)

# Train and evaluate
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=78)
model = train_logistic_regression(X_train, y_train)
results = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {results['accuracy']:.2%}")
```

### Running the Jupyter Notebook

```bash
pip install jupyter seaborn matplotlib
jupyter notebook "Evaluation Project 6 Loan Application Status Prediction-Copy1.ipynb"
```

## Running Tests

```bash
# Run all tests with coverage report
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Output:
# 75 passed
# src/data_preprocessing.py    100%
# src/feature_engineering.py   100%
# src/model.py                 100%
# TOTAL                        100%
```

## Key Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~82% |
| Decision Tree (tuned) | ~79% |

> Credit History is the strongest predictor of loan approval (correlation: 0.56).

## Tech Stack

- **Python** -- Core language
- **Pandas & NumPy** -- Data manipulation
- **Scikit-Learn** -- ML models, preprocessing, evaluation
- **SciPy** -- Statistical transformations
- **Matplotlib & Seaborn** -- Visualization (notebook)
- **Pytest** -- Unit testing framework

## License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">
Made by <a href="https://github.com/Ritesh200508">Ritesh</a>
</div>

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(filepath):
    """Load the loan dataset from a CSV/XLS file."""
    df = pd.read_csv(filepath)
    return df


def check_duplicates(df):
    """Return the number of duplicate rows in the dataframe."""
    return df.duplicated().sum()


def get_missing_value_summary(df):
    """Return a Series with the count of missing values per column."""
    return df.isnull().sum()


def drop_id_column(df):
    """Drop the Loan_ID column if present."""
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)
    return df


def impute_missing_values(df):
    """Impute missing values using domain-specific strategies.

    - Credit_History, Self_Employed, Dependents, Married: mode
    - Gender: 'unknown'
    - Loan_Amount_Term: mode
    - LoanAmount: median
    """
    df = df.copy()

    mode_columns = ["Credit_History", "Self_Employed", "Dependents", "Married"]
    for col in mode_columns:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    if "Gender" in df.columns and df["Gender"].isnull().any():
        df["Gender"] = df["Gender"].fillna("unknown")

    if "Loan_Amount_Term" in df.columns and df["Loan_Amount_Term"].isnull().any():
        df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(
            df["Loan_Amount_Term"].mode()[0]
        )

    if "LoanAmount" in df.columns and df["LoanAmount"].isnull().any():
        df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

    return df


def convert_dependents(df):
    """Convert Dependents column: replace '3+' with 3 and cast to int."""
    df = df.copy()
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace("3+", 3)
        df["Dependents"] = df["Dependents"].astype(int)
    return df


def encode_categorical(df):
    """Label-encode all object-type columns. Returns (encoded_df, encoders_dict)."""
    df = df.copy()
    encoders = {}
    category_cols = df.select_dtypes(include="object").columns
    for col in category_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def preprocess_pipeline(df):
    """Run the full preprocessing pipeline on raw data.

    Returns (processed_df, encoders).
    """
    df = drop_id_column(df)
    df = impute_missing_values(df)
    df = convert_dependents(df)
    df, encoders = encode_categorical(df)
    return df, encoders

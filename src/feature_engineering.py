import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import PowerTransformer, StandardScaler


def remove_outliers_zscore(df, threshold=3):
    """Remove rows where any column has a z-score above the threshold.

    Returns (cleaned_df, num_removed).
    """
    z = np.abs(zscore(df.select_dtypes(include=[np.number])))
    mask = (z < threshold).all(axis=1)
    cleaned = df[mask].copy()
    num_removed = len(df) - len(cleaned)
    return cleaned, num_removed


def compute_skewness(df, columns=None):
    """Compute skewness for the specified columns (or all numeric columns)."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[columns].skew()


def transform_skewed_features(df, columns, method="yeo-johnson"):
    """Apply PowerTransformer to reduce skewness in specified columns.

    Returns (transformed_df, fitted_transformer).
    """
    df = df.copy()
    transformer = PowerTransformer(method=method)
    df[columns] = transformer.fit_transform(df[columns].values)
    return df, transformer


def scale_features(X):
    """Apply StandardScaler to feature matrix X.

    Returns (scaled_array, fitted_scaler).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def split_features_target(df, target_col="Loan_Status"):
    """Split dataframe into feature matrix X and target Series y."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y


def feature_engineering_pipeline(df, skew_columns=None, target_col="Loan_Status"):
    """Run the full feature engineering pipeline.

    Steps: outlier removal -> skewness transform -> split -> scale.
    Returns (X_scaled, y, scaler, transformer, num_outliers_removed).
    """
    df, num_removed = remove_outliers_zscore(df)

    if skew_columns is None:
        skew_columns = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
    existing_skew_cols = [c for c in skew_columns if c in df.columns]
    transformer = None
    if existing_skew_cols:
        df, transformer = transform_skewed_features(df, existing_skew_cols)

    X, y = split_features_target(df, target_col)
    X_scaled, scaler = scale_features(X)

    return X_scaled, y, scaler, transformer, num_removed

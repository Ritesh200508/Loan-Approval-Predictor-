"""Shared utilities for the Loan Approval Predictor pipeline.

Consolidates duplicated data-loading, preprocessing, visualization,
and model-evaluation patterns used across the notebook.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data loading & inspection
# ---------------------------------------------------------------------------

def load_data(filepath):
    """Load dataset and print basic shape info."""
    df = pd.read_csv(filepath)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    return df


def inspect_data(df):
    """Print info, describe, and missing-value summary in one call."""
    df.info()
    print("\n--- Descriptive Statistics ---")
    print(df.describe())
    print("\n--- Missing Values ---")
    print(missing_value_summary(df))


def missing_value_summary(df):
    """Return a DataFrame with missing-value counts and percentages."""
    missing = df.isnull().sum().sort_values(ascending=False)
    pct = (missing / len(df)) * 100
    return pd.concat(
        [missing, pct],
        axis=1,
        keys=["Missing Values", "% Missing"],
    )


# ---------------------------------------------------------------------------
# Missing-value imputation
# ---------------------------------------------------------------------------

def impute_missing(df, mode_cols=None, median_cols=None, fill_values=None):
    """Impute missing values in-place.

    Parameters
    ----------
    df : DataFrame
    mode_cols : list[str] | None
        Columns to fill with their mode.
    median_cols : list[str] | None
        Columns to fill with their median.
    fill_values : dict[str, Any] | None
        Column→value pairs for custom fills (e.g. ``{'Gender': 'unknown'}``).
    """
    for col in mode_cols or []:
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in median_cols or []:
        df[col].fillna(df[col].median(), inplace=True)
    for col, val in (fill_values or {}).items():
        df[col].fillna(val, inplace=True)


# ---------------------------------------------------------------------------
# Encoding & scaling
# ---------------------------------------------------------------------------

def encode_categoricals(df):
    """Label-encode all object columns in *df* (in-place) and return encoder."""
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])
    return le


def scale_features(X):
    """StandardScaler fit-transform; returns (scaled_array, scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def remove_skewness(df, columns, method="yeo-johnson"):
    """Apply PowerTransformer to *columns* in-place."""
    pt = PowerTransformer(method=method)
    df[columns] = pt.fit_transform(df[columns].values)
    return pt


# ---------------------------------------------------------------------------
# Outlier removal
# ---------------------------------------------------------------------------

def remove_outliers_zscore(df, threshold=3):
    """Remove rows where any column's absolute z-score exceeds *threshold*.

    Returns the cleaned DataFrame and prints data-loss statistics.
    """
    z = np.abs(zscore(df))
    clean = df[(z < threshold).all(axis=1)].copy()
    lost = df.shape[0] - clean.shape[0]
    pct = lost / df.shape[0] * 100
    print(f"Before: {df.shape[0]} rows | After: {clean.shape[0]} rows | "
          f"Removed: {lost} ({pct:.1f}%)")
    return clean


# ---------------------------------------------------------------------------
# Feature / target splitting
# ---------------------------------------------------------------------------

def split_features_target(df, target_col="Loan_Status"):
    """Split DataFrame into feature matrix X and target Series Y."""
    X = df.drop(columns=[target_col])
    Y = df[target_col]
    return X, Y


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_countplots(df, columns, ncols=3, figsize=(20, 20), palette="gist_rainbow_r"):
    """Countplots for a list of categorical columns."""
    sns.set_palette(palette)
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor="white")
    axes = np.array(axes).flatten()
    for idx, col in enumerate(columns):
        sns.countplot(x=df[col], ax=axes[idx])
        axes[idx].set_xlabel(col, fontsize=20)
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_boxplots(df, columns, ncols=2, figsize=(12, 8), color="c"):
    """Boxplots for a list of numerical columns."""
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor="white")
    axes = np.array(axes).flatten()
    for idx, col in enumerate(columns):
        sns.boxplot(y=df[col], ax=axes[idx], color=color)
        axes[idx].set_xlabel(col, fontsize=20)
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_distributions(df, columns, ncols=4, figsize=(22, 5), color="r"):
    """Distribution plots for a list of numerical columns."""
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor="white")
    axes = np.array(axes).flatten()
    for idx, col in enumerate(columns):
        sns.distplot(df[col], ax=axes[idx], color=color)
        axes[idx].set_xlabel(col, fontsize=20)
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_box_and_dist(df, column, figsize=(14, 6)):
    """Side-by-side boxplot + distribution plot for a single column."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    sns.boxplot(y=df[column], ax=ax1, color="cyan")
    ax1.set_ylabel(column, fontsize=15)
    sns.distplot(df[column], ax=ax2, color="b")
    ax2.set_xlabel(column, fontsize=15)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Model training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(model, X_train, X_test, Y_train, Y_test, verbose=True):
    """Fit *model*, predict on test set, and print evaluation metrics.

    Returns
    -------
    dict with keys: model, y_pred, train_score, test_score
    """
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    train_score = model.score(X_train, Y_train)
    test_score = model.score(X_test, Y_test)
    if verbose:
        print(f"Train accuracy: {train_score:.4f}")
        print(f"Test  accuracy: {test_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(Y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(Y_test, y_pred))
    return {
        "model": model,
        "y_pred": y_pred,
        "train_score": train_score,
        "test_score": test_score,
    }


def find_best_random_state(model_class, X, Y, test_size=0.3, n_iter=250):
    """Brute-force search over random_state values for the best accuracy."""
    best_acc, best_rs = 0, 0
    for rs in range(1, n_iter + 1):
        X_tr, X_te, Y_tr, Y_te = train_test_split(
            X, Y, test_size=test_size, random_state=rs,
        )
        model = model_class()
        model.fit(X_tr, Y_tr)
        acc = accuracy_score(Y_te, model.predict(X_te))
        if acc > best_acc:
            best_acc, best_rs = acc, rs
    print(f"Best accuracy: {best_acc:.4f} at random_state={best_rs}")
    return best_rs, best_acc


def grid_search_evaluate(
    model, param_grid, X_train, X_test, Y_train, Y_test, cv=5
):
    """Run GridSearchCV, refit with best params, and evaluate.

    Returns
    -------
    dict with keys: best_params, grid_search, results
    """
    gs = GridSearchCV(model, param_grid, cv=cv)
    gs.fit(X_train, Y_train)
    print(f"Best params: {gs.best_params_}")
    best_model = gs.best_estimator_
    results = train_and_evaluate(best_model, X_train, X_test, Y_train, Y_test)
    return {"best_params": gs.best_params_, "grid_search": gs, "results": results}

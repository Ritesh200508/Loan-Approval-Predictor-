"""
Loan Approval Predictor - Production-ready module with proper error handling.

This module refactors the notebook pipeline into a structured, error-handling-aware
implementation. Each stage validates its inputs and propagates meaningful errors
instead of silently swallowing them.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Configure logging instead of silencing warnings globally
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when the dataset cannot be loaded or parsed."""


class DataValidationError(Exception):
    """Raised when data fails validation checks after processing."""


class ModelTrainingError(Exception):
    """Raised when model training fails."""


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the loan prediction dataset with proper error handling.

    Args:
        filepath: Path to the dataset file.

    Returns:
        DataFrame containing the raw dataset.

    Raises:
        DataLoadError: If the file is not found, is empty, or cannot be parsed.
    """
    path = Path(filepath)

    if not path.exists():
        raise DataLoadError(
            f"Dataset file not found: '{filepath}'. "
            f"Ensure the file exists in the working directory: {Path.cwd()}"
        )

    if path.stat().st_size == 0:
        raise DataLoadError(f"Dataset file is empty: '{filepath}'")

    try:
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError as e:
        raise DataLoadError(f"No data to parse in '{filepath}': {e}") from e
    except pd.errors.ParserError as e:
        raise DataLoadError(
            f"Failed to parse '{filepath}' as CSV. "
            f"Check file format and encoding: {e}"
        ) from e
    except UnicodeDecodeError as e:
        raise DataLoadError(
            f"Encoding error reading '{filepath}'. Try specifying encoding: {e}"
        ) from e

    if df.empty:
        raise DataLoadError(f"Dataset loaded from '{filepath}' contains no rows.")

    expected_columns = {
        "Loan_ID", "Gender", "Married", "Dependents", "Education",
        "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
        "LoanAmount", "Loan_Amount_Term", "Credit_History",
        "Property_Area", "Loan_Status",
    }
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise DataLoadError(
            f"Dataset is missing required columns: {missing_columns}. "
            f"Found columns: {list(df.columns)}"
        )

    logger.info(
        "Dataset loaded successfully: %d rows, %d columns", df.shape[0], df.shape[1]
    )
    return df


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Check for and report duplicate rows.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with duplicates removed (if any).
    """
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        logger.warning(
            "Found %d duplicate rows (%.1f%% of data). Removing duplicates.",
            n_duplicates,
            100 * n_duplicates / len(df),
        )
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        logger.info("No duplicate rows found.")
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with proper validation.

    Categorical features are filled with mode, numerical features with median.
    Raises errors if imputation cannot be performed safely.

    Args:
        df: DataFrame with potential missing values.

    Returns:
        DataFrame with missing values imputed.

    Raises:
        DataValidationError: If a column is entirely null (mode/median undefined)
            or if imputation leaves residual nulls.
    """
    categorical_impute = {
        "Gender": "mode",
        "Married": "mode",
        "Dependents": "mode",
        "Self_Employed": "mode",
        "Credit_History": "mode",
    }
    numerical_impute = {
        "LoanAmount": "median",
        "Loan_Amount_Term": "mode",
    }

    for col, strategy in categorical_impute.items():
        if col not in df.columns:
            logger.warning("Column '%s' not found in DataFrame, skipping imputation.", col)
            continue

        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue

        if df[col].isnull().all():
            raise DataValidationError(
                f"Column '{col}' is entirely null — cannot compute {strategy}."
            )

        if strategy == "mode":
            mode_values = df[col].mode()
            if mode_values.empty:
                raise DataValidationError(
                    f"Cannot compute mode for column '{col}': no non-null values."
                )
            fill_value = mode_values[0]
        else:
            fill_value = df[col].median()

        df[col] = df[col].fillna(fill_value)
        logger.info(
            "Imputed %d missing values in '%s' with %s: %s",
            n_missing, col, strategy, fill_value,
        )

    for col, strategy in numerical_impute.items():
        if col not in df.columns:
            logger.warning("Column '%s' not found in DataFrame, skipping imputation.", col)
            continue

        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue

        if df[col].isnull().all():
            raise DataValidationError(
                f"Column '{col}' is entirely null — cannot compute {strategy}."
            )

        if strategy == "median":
            fill_value = df[col].median()
        elif strategy == "mode":
            mode_values = df[col].mode()
            if mode_values.empty:
                raise DataValidationError(
                    f"Cannot compute mode for column '{col}': no non-null values."
                )
            fill_value = mode_values[0]
        else:
            fill_value = df[col].mean()

        df[col] = df[col].fillna(fill_value)
        logger.info(
            "Imputed %d missing values in '%s' with %s: %s",
            n_missing, col, strategy, fill_value,
        )

    # Validate no residual nulls in critical columns
    remaining_nulls = df[list(categorical_impute) + list(numerical_impute)].isnull().sum()
    cols_with_nulls = remaining_nulls[remaining_nulls > 0]
    if not cols_with_nulls.empty:
        raise DataValidationError(
            f"Imputation incomplete — residual nulls remain: "
            f"{cols_with_nulls.to_dict()}"
        )

    return df


def encode_dependents(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Dependents column from string to integer.

    Args:
        df: DataFrame with 'Dependents' column.

    Returns:
        DataFrame with 'Dependents' as integer type.

    Raises:
        DataValidationError: If unexpected values are found that cannot be converted.
    """
    if "Dependents" not in df.columns:
        raise DataValidationError("Column 'Dependents' not found in DataFrame.")

    # Replace '3+' with 3
    df["Dependents"] = df["Dependents"].replace("3+", 3)

    try:
        df["Dependents"] = pd.to_numeric(df["Dependents"], errors="raise").astype(int)
    except (ValueError, TypeError) as e:
        unique_vals = df["Dependents"].unique()
        raise DataValidationError(
            f"Cannot convert 'Dependents' to integer. "
            f"Unexpected values found: {unique_vals}. Error: {e}"
        ) from e

    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features using LabelEncoder with validation.

    Args:
        df: DataFrame with categorical columns.

    Returns:
        DataFrame with encoded categorical columns.

    Raises:
        DataValidationError: If encoding fails for any column.
    """
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    if not categorical_cols:
        logger.info("No categorical columns to encode.")
        return df

    le = LabelEncoder()
    for col in categorical_cols:
        n_unique = df[col].nunique()
        if n_unique == 0:
            raise DataValidationError(
                f"Column '{col}' has no valid values to encode."
            )

        # Check for nulls before encoding (LabelEncoder doesn't handle NaN)
        if df[col].isnull().any():
            raise DataValidationError(
                f"Column '{col}' contains null values. "
                f"Impute missing values before encoding."
            )

        try:
            df[col] = le.fit_transform(df[col])
        except Exception as e:
            raise DataValidationError(
                f"Failed to encode column '{col}': {e}"
            ) from e

        logger.info("Encoded '%s': %d unique categories.", col, n_unique)

    return df


def remove_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Remove outliers using z-score method with data loss validation.

    Args:
        df: DataFrame with numerical features.
        threshold: Z-score threshold for outlier detection.

    Returns:
        DataFrame with outliers removed.

    Raises:
        DataValidationError: If outlier removal would discard more than 20% of data
            or leave fewer than 30 rows.
    """
    original_shape = df.shape[0]

    try:
        z_scores = np.abs(zscore(df.select_dtypes(include=[np.number])))
    except Exception as e:
        raise DataValidationError(
            f"Failed to compute z-scores for outlier detection: {e}"
        ) from e

    mask = (z_scores < threshold).all(axis=1)
    df_cleaned = df[mask].copy()

    n_removed = original_shape - df_cleaned.shape[0]
    pct_removed = 100 * n_removed / original_shape

    if df_cleaned.shape[0] < 30:
        raise DataValidationError(
            f"Outlier removal would leave only {df_cleaned.shape[0]} rows "
            f"(removed {n_removed}). This is insufficient for model training."
        )

    if pct_removed > 20:
        raise DataValidationError(
            f"Outlier removal would discard {pct_removed:.1f}% of data "
            f"({n_removed} rows). Threshold may be too aggressive. "
            f"Consider increasing the z-score threshold from {threshold}."
        )

    logger.info(
        "Removed %d outliers (%.1f%% of data). Rows: %d -> %d",
        n_removed, pct_removed, original_shape, df_cleaned.shape[0],
    )
    return df_cleaned


def handle_skewness(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Apply Yeo-Johnson transformation to reduce skewness.

    Args:
        df: DataFrame containing the columns to transform.
        columns: List of column names to transform.

    Returns:
        DataFrame with skewness-corrected columns.

    Raises:
        DataValidationError: If transformation fails or produces invalid values.
    """
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"Columns not found for skewness transformation: {missing_cols}"
        )

    transformer = PowerTransformer(method="yeo-johnson")

    try:
        transformed = transformer.fit_transform(df[columns].values)
    except ValueError as e:
        raise DataValidationError(
            f"Yeo-Johnson transformation failed: {e}. "
            f"Check for infinite or constant-value columns."
        ) from e

    if np.any(np.isnan(transformed)) or np.any(np.isinf(transformed)):
        raise DataValidationError(
            "Yeo-Johnson transformation produced NaN or Inf values. "
            "Input data may contain problematic values."
        )

    df[columns] = transformed
    logger.info("Applied Yeo-Johnson transformation to: %s", columns)
    return df


def scale_features(X: np.ndarray) -> np.ndarray:
    """Scale features using StandardScaler with validation.

    Args:
        X: Feature matrix to scale.

    Returns:
        Scaled feature matrix.

    Raises:
        DataValidationError: If scaling produces invalid values.
    """
    scaler = StandardScaler()

    try:
        X_scaled = scaler.fit_transform(X)
    except ValueError as e:
        raise DataValidationError(f"Feature scaling failed: {e}") from e

    if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
        raise DataValidationError(
            "Feature scaling produced NaN or Inf values. "
            "Check input data for constant or problematic columns."
        )

    return X_scaled


def find_best_random_state(
    X: np.ndarray, Y: np.ndarray, max_states: int = 250
) -> tuple:
    """Find the best random state for train/test split.

    Args:
        X: Feature matrix.
        Y: Target vector.
        max_states: Number of random states to try.

    Returns:
        Tuple of (best_accuracy, best_random_state).

    Raises:
        ModelTrainingError: If no valid model could be trained.
    """
    best_accuracy = 0.0
    best_state = 0
    errors = []

    for i in range(1, max_states):
        try:
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.3, random_state=i
            )
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(Y_test, y_pred)

            if acc > best_accuracy:
                best_accuracy = acc
                best_state = i
        except Exception as e:
            errors.append((i, str(e)))
            continue

    if best_accuracy == 0.0:
        raise ModelTrainingError(
            f"Could not train any valid model across {max_states} random states. "
            f"Sample errors: {errors[:5]}"
        )

    if errors:
        logger.warning(
            "%d/%d random states produced training errors. Sample: %s",
            len(errors), max_states, errors[:3],
        )

    logger.info("Best accuracy: %.4f at random_state=%d", best_accuracy, best_state)
    return best_accuracy, best_state


def train_logistic_regression(
    X: np.ndarray, Y: np.ndarray, random_state: int = 78
) -> dict:
    """Train a Logistic Regression model with error handling.

    Args:
        X: Scaled feature matrix.
        Y: Target vector.
        random_state: Random state for reproducibility.

    Returns:
        Dict with model, predictions, and metrics.

    Raises:
        ModelTrainingError: If training or prediction fails.
    """
    try:
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, random_state=random_state, test_size=0.3
        )
    except ValueError as e:
        raise ModelTrainingError(f"Train/test split failed: {e}") from e

    model = LogisticRegression(max_iter=1000)

    try:
        model.fit(X_train, Y_train)
    except Exception as e:
        raise ModelTrainingError(
            f"Logistic Regression training failed: {e}"
        ) from e

    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        raise ModelTrainingError(f"Prediction failed: {e}") from e

    return {
        "model": model,
        "y_pred": y_pred,
        "Y_test": Y_test,
        "X_train": X_train,
        "X_test": X_test,
        "Y_train": Y_train,
        "accuracy": accuracy_score(Y_test, y_pred),
        "report": classification_report(Y_test, y_pred),
        "confusion_matrix": confusion_matrix(Y_test, y_pred),
    }


def train_decision_tree_with_grid_search(
    X_train: np.ndarray, Y_train: np.ndarray,
    X_test: np.ndarray, Y_test: np.ndarray,
) -> dict:
    """Train a Decision Tree with GridSearchCV and proper error handling.

    Args:
        X_train: Training feature matrix.
        Y_train: Training target vector.
        X_test: Test feature matrix.
        Y_test: Test target vector.

    Returns:
        Dict with best model, predictions, and metrics.

    Raises:
        ModelTrainingError: If grid search or training fails.
    """
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7, 9],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    dtc = DecisionTreeClassifier()

    try:
        grid_search = GridSearchCV(dtc, param_grid, cv=5, error_score="raise")
        grid_search.fit(X_train, Y_train)
    except Exception as e:
        raise ModelTrainingError(
            f"GridSearchCV for Decision Tree failed: {e}"
        ) from e

    best_params = grid_search.best_params_
    logger.info("Best Decision Tree params: %s", best_params)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    return {
        "model": best_model,
        "best_params": best_params,
        "y_pred": y_pred,
        "accuracy": accuracy_score(Y_test, y_pred),
        "report": classification_report(Y_test, y_pred),
        "confusion_matrix": confusion_matrix(Y_test, y_pred),
    }


def run_pipeline(filepath: str) -> dict:
    """Run the full loan prediction pipeline.

    This is the main entry point that orchestrates all steps with
    proper error propagation at each stage.

    Args:
        filepath: Path to the dataset CSV file.

    Returns:
        Dict containing trained models and evaluation metrics.

    Raises:
        DataLoadError: If data cannot be loaded.
        DataValidationError: If data fails validation at any stage.
        ModelTrainingError: If model training fails.
    """
    # Stage 1: Load data
    logger.info("=" * 60)
    logger.info("STAGE 1: Loading dataset")
    logger.info("=" * 60)
    df = load_dataset(filepath)

    # Stage 2: Data integrity
    logger.info("=" * 60)
    logger.info("STAGE 2: Data integrity check")
    logger.info("=" * 60)
    df = check_duplicates(df)

    # Drop Loan_ID (not a predictive feature)
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    # Stage 3: Missing value imputation
    logger.info("=" * 60)
    logger.info("STAGE 3: Missing value imputation")
    logger.info("=" * 60)
    df = impute_missing_values(df)

    # Stage 4: Encode features
    logger.info("=" * 60)
    logger.info("STAGE 4: Feature encoding")
    logger.info("=" * 60)
    df = encode_dependents(df)
    df = encode_categorical_features(df)

    # Stage 5: Outlier removal
    logger.info("=" * 60)
    logger.info("STAGE 5: Outlier removal")
    logger.info("=" * 60)
    df = remove_outliers(df, threshold=3.0)

    # Stage 6: Skewness correction
    logger.info("=" * 60)
    logger.info("STAGE 6: Skewness correction")
    logger.info("=" * 60)
    skew_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
    df = handle_skewness(df, skew_cols)

    # Stage 7: Prepare features and target
    logger.info("=" * 60)
    logger.info("STAGE 7: Feature preparation")
    logger.info("=" * 60)
    if "Loan_Status" not in df.columns:
        raise DataValidationError("Target column 'Loan_Status' not found after processing.")

    X = df.drop("Loan_Status", axis=1)
    Y = df["Loan_Status"]

    if len(Y.unique()) < 2:
        raise DataValidationError(
            f"Target variable has only {len(Y.unique())} unique value(s). "
            "Need at least 2 classes for classification."
        )

    X_scaled = scale_features(X.values)

    # Stage 8: Model training
    logger.info("=" * 60)
    logger.info("STAGE 8: Model training")
    logger.info("=" * 60)

    logger.info("Training Logistic Regression...")
    lr_results = train_logistic_regression(X_scaled, Y)
    logger.info("Logistic Regression accuracy: %.4f", lr_results["accuracy"])

    logger.info("Training Decision Tree with GridSearch...")
    dt_results = train_decision_tree_with_grid_search(
        lr_results["X_train"], lr_results["Y_train"],
        lr_results["X_test"], lr_results["Y_test"],
    )
    logger.info("Decision Tree accuracy: %.4f", dt_results["accuracy"])

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    return {
        "logistic_regression": lr_results,
        "decision_tree": dt_results,
        "processed_data": df,
    }


if __name__ == "__main__":
    try:
        results = run_pipeline("loan_prediction.xls")
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"\nLogistic Regression Accuracy: {results['logistic_regression']['accuracy']:.4f}")
        print(f"Decision Tree Accuracy: {results['decision_tree']['accuracy']:.4f}")
        print(f"\nBest Decision Tree Params: {results['decision_tree']['best_params']}")
    except DataLoadError as e:
        logger.error("Data loading failed: %s", e)
        sys.exit(1)
    except DataValidationError as e:
        logger.error("Data validation failed: %s", e)
        sys.exit(2)
    except ModelTrainingError as e:
        logger.error("Model training failed: %s", e)
        sys.exit(3)
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        sys.exit(99)

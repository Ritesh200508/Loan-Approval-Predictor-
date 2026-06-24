import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import PowerTransformer, StandardScaler

from src.feature_engineering import (
    remove_outliers_zscore,
    compute_skewness,
    transform_skewed_features,
    scale_features,
    split_features_target,
    feature_engineering_pipeline,
)


@pytest.fixture
def numeric_df():
    """A small numeric dataframe with a known outlier."""
    np.random.seed(42)
    data = np.random.randn(100, 3)
    data[0, 0] = 50  # extreme outlier
    return pd.DataFrame(data, columns=["A", "B", "C"])


@pytest.fixture
def skewed_df():
    """Dataframe with positively skewed columns."""
    np.random.seed(42)
    return pd.DataFrame({
        "ApplicantIncome": np.random.exponential(5000, 100),
        "CoapplicantIncome": np.random.exponential(2000, 100),
        "LoanAmount": np.random.exponential(100, 100),
        "Loan_Status": np.random.choice([0, 1], 100),
    })


class TestRemoveOutliersZscore:
    def test_removes_outlier_row(self, numeric_df):
        cleaned, num_removed = remove_outliers_zscore(numeric_df, threshold=3)
        assert num_removed >= 1
        assert len(cleaned) < len(numeric_df)

    def test_returns_dataframe(self, numeric_df):
        cleaned, _ = remove_outliers_zscore(numeric_df)
        assert isinstance(cleaned, pd.DataFrame)

    def test_no_outliers_when_threshold_high(self, numeric_df):
        cleaned, num_removed = remove_outliers_zscore(numeric_df, threshold=100)
        assert num_removed == 0
        assert len(cleaned) == len(numeric_df)

    def test_preserves_columns(self, numeric_df):
        cleaned, _ = remove_outliers_zscore(numeric_df)
        assert list(cleaned.columns) == list(numeric_df.columns)

    def test_empty_df(self):
        df = pd.DataFrame({"A": pd.Series(dtype=float), "B": pd.Series(dtype=float)})
        cleaned, num_removed = remove_outliers_zscore(df)
        assert len(cleaned) == 0
        assert num_removed == 0


class TestComputeSkewness:
    def test_returns_series(self, skewed_df):
        result = compute_skewness(skewed_df, columns=["ApplicantIncome", "LoanAmount"])
        assert isinstance(result, pd.Series)

    def test_skewness_is_positive_for_exponential(self, skewed_df):
        result = compute_skewness(skewed_df, columns=["ApplicantIncome"])
        assert result["ApplicantIncome"] > 0

    def test_default_columns_all_numeric(self, skewed_df):
        result = compute_skewness(skewed_df)
        assert len(result) == 4  # all numeric columns

    def test_specific_columns(self, skewed_df):
        result = compute_skewness(skewed_df, columns=["LoanAmount"])
        assert len(result) == 1
        assert "LoanAmount" in result.index


class TestTransformSkewedFeatures:
    def test_reduces_skewness(self, skewed_df):
        cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
        original_skew = skewed_df[cols].skew().abs().mean()
        transformed, _ = transform_skewed_features(skewed_df, cols)
        new_skew = transformed[cols].skew().abs().mean()
        assert new_skew < original_skew

    def test_returns_transformer(self, skewed_df):
        cols = ["ApplicantIncome"]
        _, transformer = transform_skewed_features(skewed_df, cols)
        assert isinstance(transformer, PowerTransformer)

    def test_shape_preserved(self, skewed_df):
        cols = ["ApplicantIncome", "LoanAmount"]
        transformed, _ = transform_skewed_features(skewed_df, cols)
        assert transformed.shape == skewed_df.shape

    def test_does_not_mutate_input(self, skewed_df):
        original = skewed_df.copy()
        transform_skewed_features(skewed_df, ["ApplicantIncome"])
        pd.testing.assert_frame_equal(skewed_df, original)

    def test_untouched_columns_unchanged(self, skewed_df):
        cols = ["ApplicantIncome"]
        transformed, _ = transform_skewed_features(skewed_df, cols)
        pd.testing.assert_series_equal(
            transformed["Loan_Status"], skewed_df["Loan_Status"]
        )


class TestScaleFeatures:
    def test_scaled_mean_near_zero(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        scaled, _ = scale_features(X)
        assert np.allclose(scaled.mean(axis=0), 0, atol=1e-10)

    def test_scaled_std_near_one(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        scaled, _ = scale_features(X)
        assert np.allclose(scaled.std(axis=0, ddof=0), 1, atol=1e-10)

    def test_returns_scaler(self):
        X = np.array([[1], [2], [3]])
        _, scaler = scale_features(X)
        assert isinstance(scaler, StandardScaler)

    def test_shape_preserved(self):
        X = np.random.randn(50, 5)
        scaled, _ = scale_features(X)
        assert scaled.shape == X.shape


class TestSplitFeaturesTarget:
    def test_correct_split(self, skewed_df):
        X, y = split_features_target(skewed_df, "Loan_Status")
        assert "Loan_Status" not in X.columns
        assert len(y) == len(skewed_df)
        assert X.shape[1] == skewed_df.shape[1] - 1

    def test_target_values(self, skewed_df):
        _, y = split_features_target(skewed_df, "Loan_Status")
        assert set(y.unique()).issubset({0, 1})

    def test_missing_target_raises(self, skewed_df):
        with pytest.raises(KeyError):
            split_features_target(skewed_df, "NonexistentColumn")


@pytest.fixture
def pipeline_df():
    """A larger numeric dataframe suitable for the full pipeline."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "ApplicantIncome": np.random.normal(5000, 1000, n),
        "CoapplicantIncome": np.random.normal(1500, 500, n),
        "LoanAmount": np.random.normal(150, 40, n),
        "Loan_Amount_Term": np.random.choice([120, 180, 240, 360], n),
        "Credit_History": np.random.choice([0, 1], n),
        "Gender": np.random.choice([0, 1], n),
        "Married": np.random.choice([0, 1], n),
        "Dependents": np.random.choice([0, 1, 2, 3], n),
        "Education": np.random.choice([0, 1], n),
        "Self_Employed": np.random.choice([0, 1], n),
        "Property_Area": np.random.choice([0, 1, 2], n),
        "Loan_Status": np.random.choice([0, 1], n),
    })


class TestFeatureEngineeringPipeline:
    def test_end_to_end(self, pipeline_df):
        X_scaled, y, scaler, transformer, num_removed = feature_engineering_pipeline(
            pipeline_df
        )
        assert isinstance(X_scaled, np.ndarray)
        assert len(y) == X_scaled.shape[0]
        assert isinstance(scaler, StandardScaler)

    def test_outliers_removed(self, pipeline_df):
        _, _, _, _, num_removed = feature_engineering_pipeline(pipeline_df)
        assert isinstance(num_removed, int)
        assert num_removed >= 0

    def test_scaled_output_shape(self, pipeline_df):
        X_scaled, y, _, _, _ = feature_engineering_pipeline(pipeline_df)
        expected_features = pipeline_df.shape[1] - 1  # minus target
        assert X_scaled.shape[1] == expected_features

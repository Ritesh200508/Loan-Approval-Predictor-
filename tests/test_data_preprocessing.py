import numpy as np
import pandas as pd
import pytest
from src.data_preprocessing import (
    load_data,
    check_duplicates,
    get_missing_value_summary,
    drop_id_column,
    impute_missing_values,
    convert_dependents,
    encode_categorical,
    preprocess_pipeline,
)


class TestLoadData:
    def test_load_returns_dataframe(self, sample_csv_path):
        df = load_data(sample_csv_path)
        assert isinstance(df, pd.DataFrame)

    def test_load_correct_shape(self, sample_csv_path):
        df = load_data(sample_csv_path)
        assert df.shape == (10, 13)

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_data(str(tmp_path / "nonexistent.csv"))


class TestCheckDuplicates:
    def test_no_duplicates(self, sample_raw_df):
        assert check_duplicates(sample_raw_df) == 0

    def test_with_duplicates(self, sample_raw_df):
        df = pd.concat([sample_raw_df, sample_raw_df.iloc[[0]]], ignore_index=True)
        assert check_duplicates(df) == 1


class TestGetMissingValueSummary:
    def test_returns_series(self, sample_raw_df):
        result = get_missing_value_summary(sample_raw_df)
        assert isinstance(result, pd.Series)

    def test_detects_missing_gender(self, sample_raw_df):
        result = get_missing_value_summary(sample_raw_df)
        assert result["Gender"] == 2

    def test_detects_missing_loan_amount(self, sample_raw_df):
        result = get_missing_value_summary(sample_raw_df)
        assert result["LoanAmount"] == 2

    def test_no_missing_in_education(self, sample_raw_df):
        result = get_missing_value_summary(sample_raw_df)
        assert result["Education"] == 0


class TestDropIdColumn:
    def test_removes_loan_id(self, sample_raw_df):
        df = drop_id_column(sample_raw_df)
        assert "Loan_ID" not in df.columns

    def test_preserves_other_columns(self, sample_raw_df):
        df = drop_id_column(sample_raw_df)
        assert "Gender" in df.columns
        assert "Loan_Status" in df.columns

    def test_no_id_column_noop(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = drop_id_column(df)
        assert list(result.columns) == ["A", "B"]

    def test_does_not_mutate_input(self, sample_raw_df):
        original_cols = list(sample_raw_df.columns)
        drop_id_column(sample_raw_df)
        assert list(sample_raw_df.columns) == original_cols


class TestImputeMissingValues:
    def test_no_missing_after_imputation(self, sample_raw_df):
        df = drop_id_column(sample_raw_df)
        result = impute_missing_values(df)
        assert result.isnull().sum().sum() == 0

    def test_gender_filled_with_unknown(self, sample_raw_df):
        df = drop_id_column(sample_raw_df)
        result = impute_missing_values(df)
        assert (result["Gender"] == "unknown").sum() == 2

    def test_loan_amount_filled_with_median(self, sample_raw_df):
        df = drop_id_column(sample_raw_df)
        median_val = df["LoanAmount"].median()
        result = impute_missing_values(df)
        filled_vals = result.loc[sample_raw_df["LoanAmount"].isnull().values, "LoanAmount"]
        assert (filled_vals == median_val).all()

    def test_credit_history_filled_with_mode(self, sample_raw_df):
        df = drop_id_column(sample_raw_df)
        mode_val = df["Credit_History"].mode()[0]
        result = impute_missing_values(df)
        assert not result["Credit_History"].isnull().any()
        filled = result.loc[
            sample_raw_df["Credit_History"].isnull().values, "Credit_History"
        ]
        assert (filled == mode_val).all()

    def test_does_not_mutate_input(self, sample_raw_df):
        df = drop_id_column(sample_raw_df)
        original_nulls = df.isnull().sum().sum()
        impute_missing_values(df)
        assert df.isnull().sum().sum() == original_nulls


class TestConvertDependents:
    def test_three_plus_replaced(self, sample_raw_df):
        df = drop_id_column(sample_raw_df)
        df = impute_missing_values(df)
        result = convert_dependents(df)
        assert "3+" not in result["Dependents"].values
        assert 3 in result["Dependents"].values

    def test_dtype_is_int(self, sample_raw_df):
        df = drop_id_column(sample_raw_df)
        df = impute_missing_values(df)
        result = convert_dependents(df)
        assert result["Dependents"].dtype in [np.int64, np.int32, int]

    def test_does_not_mutate_input(self, sample_raw_df):
        df = drop_id_column(sample_raw_df)
        df = impute_missing_values(df)
        original_dtype = df["Dependents"].dtype
        convert_dependents(df)
        assert df["Dependents"].dtype == original_dtype


class TestEncodeCategorical:
    def test_no_object_columns_after_encoding(self, sample_raw_df):
        df = drop_id_column(sample_raw_df)
        df = impute_missing_values(df)
        df = convert_dependents(df)
        encoded, _ = encode_categorical(df)
        assert len(encoded.select_dtypes(include="object").columns) == 0

    def test_returns_encoders(self, sample_raw_df):
        df = drop_id_column(sample_raw_df)
        df = impute_missing_values(df)
        df = convert_dependents(df)
        _, encoders = encode_categorical(df)
        assert isinstance(encoders, dict)
        assert len(encoders) > 0

    def test_encoded_values_are_numeric(self, sample_raw_df):
        df = drop_id_column(sample_raw_df)
        df = impute_missing_values(df)
        df = convert_dependents(df)
        encoded, _ = encode_categorical(df)
        for col in encoded.columns:
            assert pd.api.types.is_numeric_dtype(encoded[col])

    def test_does_not_mutate_input(self, sample_raw_df):
        df = drop_id_column(sample_raw_df)
        df = impute_missing_values(df)
        df = convert_dependents(df)
        original_dtypes = df.dtypes.copy()
        encode_categorical(df)
        pd.testing.assert_series_equal(df.dtypes, original_dtypes)


class TestPreprocessPipeline:
    def test_end_to_end(self, sample_raw_df):
        result, encoders = preprocess_pipeline(sample_raw_df)
        assert "Loan_ID" not in result.columns
        assert result.isnull().sum().sum() == 0
        assert len(result.select_dtypes(include="object").columns) == 0

    def test_returns_encoders_dict(self, sample_raw_df):
        _, encoders = preprocess_pipeline(sample_raw_df)
        assert isinstance(encoders, dict)

    def test_shape_preserved_rows(self, sample_raw_df):
        result, _ = preprocess_pipeline(sample_raw_df)
        assert len(result) == len(sample_raw_df)

    def test_loan_id_dropped(self, sample_raw_df):
        result, _ = preprocess_pipeline(sample_raw_df)
        assert "Loan_ID" not in result.columns
        assert result.shape[1] == sample_raw_df.shape[1] - 1

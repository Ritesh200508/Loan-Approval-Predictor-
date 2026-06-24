import os
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_raw_df():
    """A small raw dataframe mimicking the loan dataset with missing values."""
    data = {
        "Loan_ID": ["LP001", "LP002", "LP003", "LP004", "LP005",
                     "LP006", "LP007", "LP008", "LP009", "LP010"],
        "Gender": ["Male", "Female", np.nan, "Male", "Female",
                    "Male", "Male", np.nan, "Female", "Male"],
        "Married": ["Yes", "No", "Yes", np.nan, "Yes",
                     "No", "Yes", "Yes", "No", "Yes"],
        "Dependents": ["0", "1", "2", "3+", "0",
                       "1", "2", "0", np.nan, "3+"],
        "Education": ["Graduate", "Not Graduate", "Graduate", "Graduate", "Not Graduate",
                       "Graduate", "Not Graduate", "Graduate", "Graduate", "Not Graduate"],
        "Self_Employed": ["No", "Yes", "No", np.nan, "No",
                          "Yes", "No", "No", "Yes", "No"],
        "ApplicantIncome": [5849, 4583, 3000, 2583, 6000,
                            5417, 2333, 7500, 1800, 4000],
        "CoapplicantIncome": [0, 1508, 0, 2358, 0,
                              4196, 1516, 0, 3500, 1000],
        "LoanAmount": [np.nan, 128, 66, 120, 141,
                       267, 95, 158, np.nan, 110],
        "Loan_Amount_Term": [360, 360, 360, 360, 360,
                             360, 360, np.nan, 360, 360],
        "Credit_History": [1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, np.nan, 1.0, 0.0, 1.0],
        "Property_Area": ["Urban", "Rural", "Urban", "Urban", "Rural",
                          "Semiurban", "Urban", "Rural", "Semiurban", "Urban"],
        "Loan_Status": ["Y", "N", "Y", "Y", "Y",
                        "Y", "Y", "N", "N", "Y"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def preprocessed_df(sample_raw_df):
    """A dataframe that has been through preprocessing (no ID, imputed, encoded)."""
    from src.data_preprocessing import preprocess_pipeline
    df, _ = preprocess_pipeline(sample_raw_df)
    return df


@pytest.fixture
def sample_csv_path(tmp_path, sample_raw_df):
    """Write the sample dataframe to a CSV and return its path."""
    path = tmp_path / "loan_sample.csv"
    sample_raw_df.to_csv(path, index=False)
    return str(path)

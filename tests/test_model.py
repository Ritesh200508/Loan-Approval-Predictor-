import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from src.model import (
    find_best_random_state,
    train_logistic_regression,
    train_decision_tree,
    evaluate_model,
    tune_decision_tree,
)


@pytest.fixture
def classification_data():
    """Generate a simple binary classification dataset."""
    np.random.seed(42)
    X = np.random.randn(200, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def train_test_data(classification_data):
    """Split classification data into train/test."""
    X, y = classification_data
    return train_test_split(X, y, test_size=0.3, random_state=42)


class TestFindBestRandomState:
    def test_returns_tuple(self, classification_data):
        X, y = classification_data
        result = find_best_random_state(X, y, max_rs=5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_accuracy_between_zero_and_one(self, classification_data):
        X, y = classification_data
        acc, _ = find_best_random_state(X, y, max_rs=5)
        assert 0 <= acc <= 1

    def test_random_state_is_positive_int(self, classification_data):
        X, y = classification_data
        _, rs = find_best_random_state(X, y, max_rs=5)
        assert isinstance(rs, (int, np.integer))
        assert rs >= 1

    def test_small_search_range(self, classification_data):
        X, y = classification_data
        acc, rs = find_best_random_state(X, y, max_rs=3)
        assert 1 <= rs < 3


class TestTrainLogisticRegression:
    def test_returns_model(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        model = train_logistic_regression(X_train, y_train)
        assert isinstance(model, LogisticRegression)

    def test_model_can_predict(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        model = train_logistic_regression(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)
        assert set(preds).issubset({0, 1})

    def test_reasonable_accuracy(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        model = train_logistic_regression(X_train, y_train)
        acc = model.score(X_test, y_test)
        assert acc > 0.5  # better than random


class TestTrainDecisionTree:
    def test_returns_model(self, train_test_data):
        X_train, _, y_train, _ = train_test_data
        model = train_decision_tree(X_train, y_train)
        assert isinstance(model, DecisionTreeClassifier)

    def test_custom_hyperparameters(self, train_test_data):
        X_train, _, y_train, _ = train_test_data
        model = train_decision_tree(
            X_train, y_train, criterion="entropy", max_depth=3,
            min_samples_split=5, min_samples_leaf=2
        )
        assert model.criterion == "entropy"
        assert model.max_depth == 3
        assert model.min_samples_split == 5
        assert model.min_samples_leaf == 2

    def test_model_can_predict(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        model = train_decision_tree(X_train, y_train, max_depth=5)
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)

    def test_overfits_training_data_without_depth_limit(self, train_test_data):
        X_train, _, y_train, _ = train_test_data
        model = train_decision_tree(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        assert train_acc >= 0.95


class TestEvaluateModel:
    def test_returns_dict_keys(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        model = train_logistic_regression(X_train, y_train)
        result = evaluate_model(model, X_test, y_test)
        assert "accuracy" in result
        assert "confusion_matrix" in result
        assert "classification_report" in result
        assert "predictions" in result

    def test_accuracy_type(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        model = train_logistic_regression(X_train, y_train)
        result = evaluate_model(model, X_test, y_test)
        assert isinstance(result["accuracy"], float)
        assert 0 <= result["accuracy"] <= 1

    def test_confusion_matrix_shape(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        model = train_logistic_regression(X_train, y_train)
        result = evaluate_model(model, X_test, y_test)
        assert result["confusion_matrix"].shape == (2, 2)

    def test_classification_report_is_dict(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        model = train_logistic_regression(X_train, y_train)
        result = evaluate_model(model, X_test, y_test)
        assert isinstance(result["classification_report"], dict)
        assert "accuracy" in result["classification_report"]

    def test_predictions_length(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        model = train_logistic_regression(X_train, y_train)
        result = evaluate_model(model, X_test, y_test)
        assert len(result["predictions"]) == len(y_test)

    def test_evaluate_decision_tree(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        model = train_decision_tree(X_train, y_train, max_depth=3)
        result = evaluate_model(model, X_test, y_test)
        assert 0 <= result["accuracy"] <= 1


class TestTuneDecisionTree:
    def test_returns_best_model(self, train_test_data):
        X_train, _, y_train, _ = train_test_data
        param_grid = {
            "criterion": ["gini"],
            "max_depth": [3, 5],
            "min_samples_split": [2],
            "min_samples_leaf": [1],
        }
        best_model, best_params, best_score = tune_decision_tree(
            X_train, y_train, param_grid=param_grid, cv=2
        )
        assert isinstance(best_model, DecisionTreeClassifier)

    def test_best_params_is_dict(self, train_test_data):
        X_train, _, y_train, _ = train_test_data
        param_grid = {
            "criterion": ["gini"],
            "max_depth": [3],
            "min_samples_split": [2],
            "min_samples_leaf": [1],
        }
        _, best_params, _ = tune_decision_tree(
            X_train, y_train, param_grid=param_grid, cv=2
        )
        assert isinstance(best_params, dict)
        assert "criterion" in best_params
        assert "max_depth" in best_params

    def test_best_score_valid(self, train_test_data):
        X_train, _, y_train, _ = train_test_data
        param_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [3, 5],
            "min_samples_split": [2],
            "min_samples_leaf": [1],
        }
        _, _, best_score = tune_decision_tree(
            X_train, y_train, param_grid=param_grid, cv=2
        )
        assert 0 <= best_score <= 1

    def test_tuned_model_can_predict(self, train_test_data):
        X_train, X_test, y_train, _ = train_test_data
        param_grid = {
            "criterion": ["gini"],
            "max_depth": [3],
            "min_samples_split": [2],
            "min_samples_leaf": [1],
        }
        best_model, _, _ = tune_decision_tree(
            X_train, y_train, param_grid=param_grid, cv=2
        )
        preds = best_model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_default_param_grid(self, train_test_data):
        X_train, _, y_train, _ = train_test_data
        best_model, best_params, best_score = tune_decision_tree(
            X_train, y_train, cv=2
        )
        assert isinstance(best_model, DecisionTreeClassifier)
        assert "criterion" in best_params
        assert 0 <= best_score <= 1

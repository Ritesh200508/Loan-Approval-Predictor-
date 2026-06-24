import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def find_best_random_state(X, y, model_class=LogisticRegression, max_rs=250, test_size=0.3):
    """Search for the random_state that gives the highest accuracy.

    Returns (best_accuracy, best_random_state).
    """
    best_acc = 0
    best_rs = 0
    for i in range(1, max_rs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=i
        )
        model = model_class()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_rs = i
    return best_acc, best_rs


def train_logistic_regression(X_train, y_train):
    """Train a LogisticRegression model and return it."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, criterion="gini", max_depth=None,
                        min_samples_split=2, min_samples_leaf=1):
    """Train a DecisionTreeClassifier with given hyperparameters."""
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model. Returns dict with accuracy, confusion_matrix, report."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "predictions": y_pred,
    }


def tune_decision_tree(X_train, y_train, param_grid=None, cv=5):
    """Run GridSearchCV on DecisionTreeClassifier.

    Returns (best_model, best_params, best_score).
    """
    if param_grid is None:
        param_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [3, 5, 7, 9],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    dtc = DecisionTreeClassifier()
    grid_search = GridSearchCV(dtc, param_grid, cv=cv)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

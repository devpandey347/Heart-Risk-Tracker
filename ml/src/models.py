"""
models.py — Model definitions for the Heart Disease classifier comparison.
===========================================================================
Five classifiers are configured here with baseline + tuned hyperparameters.
All models are wrapped in sklearn Pipelines (preprocessor + estimator),
so they can be trained and evaluated identically.

Why these five?
  LR  — fast, interpretable, good baseline for linearly separable data
  RF  — robust, handles feature interactions, little tuning needed
  SVM — strong on small/medium datasets, needs scaling (handled in pipeline)
  KNN — simple, non-parametric; baseline for instance-based learning
  GB  — usually top performer on tabular data; gradient-boosted decision trees
"""

from typing import Dict, List, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from preprocessing import build_preprocessor


def get_base_pipeline(model_name: str) -> Pipeline:
    """Return an unfitted base Pipeline for a given model family."""
    preprocessor = build_preprocessor()

    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
            C=1.0,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="euclidean",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    if model_name not in classifiers:
        raise ValueError(f"Unknown model name: {model_name}")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifiers[model_name]),
    ])


def get_model_param_grids() -> Dict[str, List[Dict[str, Any]]]:
    """
    Return conditional parameter grids for nested CV hyperparameter tuning.
    
    Conditional grids ensure irrelevant combinations (e.g. gamma for linear SVM)
    are not evaluated.
    """
    scalers = [StandardScaler(), RobustScaler()]

    return {
        "KNN": [
            {
                "preprocessor__num__scaler": scalers,
                "classifier__n_neighbors": [5, 7, 9, 11, 13, 15],
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["euclidean"],
                "classifier__p": [2],
            },
            {
                "preprocessor__num__scaler": scalers,
                "classifier__n_neighbors": [5, 7, 9, 11, 13, 15],
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["manhattan"],
                "classifier__p": [1],
            },
        ],

        "SVM": [
            # Non-linear RBF kernel (gamma active)
            {
                "preprocessor__num__scaler": scalers,
                "classifier__kernel": ["rbf"],
                "classifier__C": [0.1, 0.5, 1.0, 2.0, 5.0],
                "classifier__gamma": ["scale", "auto", 0.05, 0.1],
                "classifier__class_weight": [None, "balanced"],
            },
            # Non-linear Sigmoid kernel (gamma active)
            {
                "preprocessor__num__scaler": scalers,
                "classifier__kernel": ["sigmoid"],
                "classifier__C": [0.1, 0.5, 1.0, 2.0, 5.0],
                "classifier__gamma": ["scale", "auto", 0.05, 0.1],
                "classifier__class_weight": [None, "balanced"],
            },
            # Linear kernel (gamma omitted to avoid invalid/redundant evaluation)
            {
                "preprocessor__num__scaler": scalers,
                "classifier__kernel": ["linear"],
                "classifier__C": [0.1, 0.5, 1.0, 2.0, 5.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ],

        "Logistic Regression": [
            {
                "preprocessor__num__scaler": scalers,
                "classifier__C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
                "classifier__class_weight": [None, "balanced"],
                "classifier__solver": ["lbfgs"],
                "classifier__max_iter": [1000],
            },
        ],

        "Random Forest": [
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 5, 8],
                "classifier__min_samples_split": [2, 4, 6],
                "classifier__random_state": [42],
            },
        ],

        "Gradient Boosting": [
            {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.02, 0.05, 0.1],
                "classifier__max_depth": [2, 3],
                "classifier__subsample": [0.8, 1.0],
                "classifier__random_state": [42],
            },
        ],
    }


def get_model_configs() -> dict[str, Pipeline]:
    """
    Return a dict of name -> baseline sklearn Pipeline.
    Maintained for backward compatibility with existing modules and tests.
    """
    model_names = [
        "Logistic Regression",
        "Random Forest",
        "SVM",
        "KNN",
        "Gradient Boosting",
    ]
    return {name: get_base_pipeline(name) for name in model_names}

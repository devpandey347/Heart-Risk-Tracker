"""
preprocessing.py — Leakage-safe sklearn ColumnTransformer for Heart Disease dataset.
======================================================================================
Imported by train_pipeline.py. Never fitted on test data.

Pipeline structure (v2 — corrected encoding)
---------------------------------------------
Numerical  (6): SimpleImputer(median)  → StandardScaler
Binary cat (3): SimpleImputer(most_frequent) → passthrough as 0/1
                  sex (0=F, 1=M), fbs (0/1), exang (0/1)
Nominal cat (4): SimpleImputer(most_frequent) → OneHotEncoder(drop='first')
                  cp (4 levels → 3 dummy cols)
                  restecg (3 levels → 2 dummy cols)
                  slope   (3 levels → 2 dummy cols)
                  thal    (3 levels → 2 dummy cols)

Total transformed features: 6 + 3 + (3+2+2+2) = 18

Why OneHotEncoder for the four multi-level nominal features?
------------------------------------------------------------
• cp, restecg, slope, and thal are NOMINAL (unordered) categories.  Applying
  OrdinalEncoder imposes a false numeric ranking (e.g. cp: 1<2<3<4) that biases
  distance-based and linear models (KNN, SVM, Logistic Regression) because they
  interpret larger ordinal codes as "more" of something, which is clinically wrong.
• RandomForest and GradientBoosting are tree-based and split on individual
  thresholds — they are unaffected by encoding choice for nominal categoricals.
• Binary features (sex 0/1, fbs 0/1, exang 0/1) are already valid 0/1 numerics
  and need no further expansion; they pass through as-is.

Feature schema (13 RAW inputs in this exact order)
---------------------------------------------------
Numerical (6) : age, trestbps, chol, thalach, oldpeak, ca
Binary cat (3): sex, fbs, exang
Nominal cat (4): cp, restecg, slope, thal
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

# ── Feature schema ──────────────────────────────────────────────────────────
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

# Binary categoricals: already 0/1 — treated as numeric passthrough
BINARY_FEATURES = ["sex", "fbs", "exang"]

# Nominal (unordered) categoricals with 3+ levels — get OneHotEncoded
NOMINAL_FEATURES = ["cp", "restecg", "slope", "thal"]

# All categorical features together (order matters for CATEGORICAL_FEATURES compat)
CATEGORICAL_FEATURES = BINARY_FEATURES + NOMINAL_FEATURES  # ["sex","fbs","exang","cp","restecg","slope","thal"]

# All 13 raw input feature names in canonical order
# NOTE: this order is different from v1 — binary features come before nominal.
# The pipeline handles column selection by name so order only affects output column layout.
ALL_FEATURES = NUMERICAL_FEATURES + ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

# Known categories for OneHotEncoder — drop='first' removes one dummy per feature
# to avoid the dummy-variable trap (perfect multicollinearity).
NOMINAL_CATEGORIES = [
    [1.0, 2.0, 3.0, 4.0],   # cp:       1=typical angina, 2=atypical, 3=non-anginal, 4=asymptomatic
    [0.0, 1.0, 2.0],        # restecg:  0=normal, 1=ST-T abnormality, 2=LV hypertrophy
    [1.0, 2.0, 3.0],        # slope:    1=upsloping, 2=flat, 3=downsloping
    [3.0, 6.0, 7.0],        # thal:     3=normal, 6=fixed defect, 7=reversible defect
]


def build_preprocessor(scaler_type: str = "standard") -> ColumnTransformer:
    """
    Build and return an unfitted ColumnTransformer.

    Parameters
    ----------
    scaler_type : 'standard' for StandardScaler, 'robust' for RobustScaler.

    Call pipeline.fit(X_train, y_train) — the pipeline will fit the
    transformer on training data only.
    """
    scaler = RobustScaler() if scaler_type.lower() == "robust" else StandardScaler()
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  scaler),
    ])

    # Binary: impute missing then pass raw 0/1 integer values through
    binary_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # No encoder needed — 0/1 is already a valid numeric representation
    ])

    # Nominal: impute then one-hot encode, dropping first level to avoid collinearity
    nominal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            categories=NOMINAL_CATEGORIES,
            drop="first",            # drop first level per feature
            sparse_output=False,     # return dense array (compat with SHAP)
            handle_unknown="ignore", # if an unknown value arrives, all dummies = 0
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",     numerical_pipeline, NUMERICAL_FEATURES),
            ("binary",  binary_pipeline,    BINARY_FEATURES),
            ("nominal", nominal_pipeline,   NOMINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,  # True so get_feature_names_out() is unambiguous
    )

    return preprocessor

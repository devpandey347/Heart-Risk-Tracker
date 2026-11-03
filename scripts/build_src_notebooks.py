"""
build_src_notebooks.py — Generates .ipynb notebooks directly in ml/src/ with relative paths suited for ml/src execution.
Synchronized with Heart Risk Tracker ML implementation, nested-CV protocol, and SVM champion.
"""

import json
from pathlib import Path

SRC_DIR = Path("ml/src")


def make_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.13",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated: {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. ml/src/eda.ipynb
# ─────────────────────────────────────────────────────────────────────────────
eda_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Exploratory Data Analysis (EDA) — Cleveland Heart Disease Dataset\n",
            "\n",
            "Interactive notebook for exploring patient distributions, missing values, and clinical correlations in `processed.cleveland.data` ($N=303$).",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys\n",
            "from pathlib import Path\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import seaborn as sns\n",
            "\n",
            "# In ml/src, modules are in the current working directory\n",
            "from data_loader import load_raw, apply_binary_target, validate_features, NUMERICAL_FEATURES\n",
            "\n",
            "sns.set_theme(style=\"whitegrid\", font_scale=1.05)\n",
            "plt.rcParams[\"figure.dpi\"] = 120\n",
            "\n",
            "data_path = Path(\"../data/raw/processed.cleveland.data\")\n",
            "df_raw = load_raw(data_path)\n",
            "print(f\"Loaded Cleveland data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns\")\n",
            "df_raw.head()",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Missing Values Check",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "missing = df_raw.isnull().sum()\n",
            "missing_cols = missing[missing > 0]\n",
            "print(\"Missing value counts in dataset:\")\n",
            "for col, cnt in missing_cols.items():\n",
            "    print(f\"  {col}: {cnt} missing ({cnt/len(df_raw)*100:.2f}%)\")\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(5, 3))\n",
            "missing_cols.plot(kind=\"bar\", color=[\"#0d9488\", \"#0284c7\"], ax=ax)\n",
            "ax.set_title(\"Missing Values by Column\")\n",
            "ax.set_ylabel(\"Count\")\n",
            "plt.show()",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Binary Target Mapping & Cohort Balance",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df = apply_binary_target(df_raw)\n",
            "counts = df[\"target_binary\"].value_counts().sort_index()\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(6, 4))\n",
            "bars = ax.bar([\"No Disease (0)\", \"Disease Present (1)\"], counts.values, color=[\"#0d9488\", \"#e11d48\"], width=0.55)\n",
            "for bar in bars:\n",
            "    h = bar.get_height()\n",
            "    pct = h / len(df) * 100\n",
            "    ax.annotate(f\"{h} ({pct:.1f}%)\", xy=(bar.get_x() + bar.get_width() / 2, h),\n",
            "                xytext=(0, 4), textcoords=\"offset points\", ha=\"center\", fontweight=\"bold\")\n",
            "\n",
            "ax.set_title(\"Target Distribution (Total N=303)\")\n",
            "ax.set_ylabel(\"Patients\")\n",
            "ax.set_ylim(0, 200)\n",
            "plt.tight_layout()\n",
            "plt.show()",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Numerical Feature Boxplots by Heart Disease Status",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(2, 3, figsize=(15, 8))\n",
            "axes = axes.flatten()\n",
            "\n",
            "for i, col in enumerate(NUMERICAL_FEATURES):\n",
            "    sns.boxplot(data=df, x=\"target_binary\", y=col, palette=[\"#0d9488\", \"#e11d48\"], ax=axes[i], width=0.4)\n",
            "    axes[i].set_title(f\"{col.upper()} by Target\", fontweight=\"bold\")\n",
            "    axes[i].set_xticklabels([\"No Disease\", \"Disease Present\"])\n",
            "    axes[i].set_xlabel(\"\")\n",
            "\n",
            "axes[5].axis(\"off\")\n",
            "plt.tight_layout()\n",
            "plt.show()",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Feature Correlation Matrix Heatmap",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, ax = plt.subplots(figsize=(10, 8))\n",
            "corr = df.corr()\n",
            "mask = np.triu(np.ones_like(corr, dtype=bool))\n",
            "sns.heatmap(corr, mask=mask, annot=True, fmt=\".2f\", cmap=\"vlag\", center=0, square=True, linewidths=0.5, ax=ax)\n",
            "ax.set_title(\"Pairwise Feature Correlation Matrix\", fontweight=\"bold\")\n",
            "plt.tight_layout()\n",
            "plt.show()",
        ],
    },
]

make_notebook(eda_cells, SRC_DIR / "eda.ipynb")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ml/src/train_pipeline.ipynb
# ─────────────────────────────────────────────────────────────────────────────
train_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Model Training & Five-Algorithm Comparison\n",
            "\n",
            "Interactive notebook for training and evaluating 5 machine learning classifiers:\n",
            "- Support Vector Machine (SVM — Champion)\n",
            "- Logistic Regression\n",
            "- K-Nearest Neighbors (KNN)\n",
            "- Gradient Boosting Classifier\n",
            "- Random Forest Classifier\n",
            "\n",
            "**Protocol**: 80/20 Stratified Split ($N=242$ train, $N=61$ test, seed 2026) with 5-fold cross-validation on training data.",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys\n",
            "from pathlib import Path\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import seaborn as sns\n",
            "from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score\n",
            "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc\n",
            "\n",
            "from data_loader import load_raw, apply_binary_target\n",
            "from preprocessing import ALL_FEATURES\n",
            "from models import get_base_pipeline\n",
            "\n",
            "sns.set_theme(style=\"whitegrid\", font_scale=1.05)\n",
            "plt.rcParams[\"figure.dpi\"] = 120\n",
            "\n",
            "data_path = Path(\"../data/raw/processed.cleveland.data\")\n",
            "df = apply_binary_target(load_raw(data_path))\n",
            "\n",
            "X = df[ALL_FEATURES]\n",
            "y = df[\"target_binary\"]\n",
            "\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2026, stratify=y)\n",
            "print(f\"Training set samples: {len(X_train)}  |  Held-out test samples: {len(X_test)}\")",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Train 5 Models with 5-Fold Stratified Cross-Validation",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "model_names = [\"SVM\", \"Logistic Regression\", \"KNN\", \"Gradient Boosting\", \"Random Forest\"]\n",
            "cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
            "\n",
            "results = []\n",
            "fitted_models = {}\n",
            "\n",
            "for name in model_names:\n",
            "    pipeline = get_base_pipeline(name)\n",
            "    cv_acc = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring=\"accuracy\")\n",
            "    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring=\"f1\")\n",
            "    \n",
            "    pipeline.fit(X_train, y_train)\n",
            "    fitted_models[name] = pipeline\n",
            "    \n",
            "    y_pred = pipeline.predict(X_test)\n",
            "    y_prob = pipeline.predict_proba(X_test)[:, 1]\n",
            "    \n",
            "    results.append({\n",
            "        \"Algorithm\": name,\n",
            "        \"CV Acc Mean\": float(cv_acc.mean()),\n",
            "        \"CV Acc Std\": float(cv_acc.std()),\n",
            "        \"CV F1 Mean\": float(cv_f1.mean()),\n",
            "        \"Test Acc\": float(accuracy_score(y_test, y_pred)),\n",
            "        \"Precision\": float(precision_score(y_test, y_pred, zero_division=0)),\n",
            "        \"Recall\": float(recall_score(y_test, y_pred, zero_division=0)),\n",
            "        \"F1\": float(f1_score(y_test, y_pred, zero_division=0)),\n",
            "        \"ROC-AUC\": float(roc_auc_score(y_test, y_prob)),\n",
            "    })\n",
            "\n",
            "df_res = pd.DataFrame(results)\n",
            "df_res[[\"Algorithm\", \"CV Acc Mean\", \"CV F1 Mean\", \"Test Acc\", \"Precision\", \"Recall\", \"F1\", \"ROC-AUC\"]]",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Grouped Bar Chart: CV Accuracy vs Held-Out Test Accuracy",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, ax = plt.subplots(figsize=(11, 5))\n",
            "x = np.arange(len(df_res))\n",
            "w = 0.35\n",
            "\n",
            "b1 = ax.bar(x - w/2, df_res[\"CV Acc Mean\"] * 100, w, label=\"CV Accuracy (Train Folds N=242)\", color=\"#0d9488\")\n",
            "b2 = ax.bar(x + w/2, df_res[\"Test Acc\"] * 100, w, label=\"Test Accuracy (Held-Out N=61, seed=2026)\", color=\"#0284c7\")\n",
            "\n",
            "ax.set_ylabel(\"Accuracy (%)\", fontweight=\"bold\")\n",
            "ax.set_title(\"5-Algorithm Accuracy Comparison\", fontsize=13, fontweight=\"bold\")\n",
            "ax.set_xticks(x)\n",
            "ax.set_xticklabels(df_res[\"Algorithm\"], rotation=15, ha=\"right\", fontweight=\"bold\")\n",
            "ax.set_ylim(60, 100)\n",
            "ax.legend()\n",
            "\n",
            "for b in b1:\n",
            "    h = b.get_height()\n",
            "    ax.annotate(f\"{h:.1f}%\", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords=\"offset points\", ha=\"center\", color=\"#0d9488\", fontweight=\"bold\")\n",
            "for b in b2:\n",
            "    h = b.get_height()\n",
            "    ax.annotate(f\"{h:.1f}%\", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords=\"offset points\", ha=\"center\", color=\"#0284c7\", fontweight=\"bold\")\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()",
        ],
    },
]

make_notebook(train_cells, SRC_DIR / "train_pipeline.ipynb")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ml/src/explainability.ipynb
# ─────────────────────────────────────────────────────────────────────────────
shap_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# SHAP Model Explainability — Heart Risk Tracker\n",
            "\n",
            "Interactive notebook for Explainable AI (XAI) using **SHAP (SHapley Additive exPlanations)** on the champion **Support Vector Machine (SVM)** model.",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys\n",
            "from pathlib import Path\n",
            "import joblib\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import shap\n",
            "\n",
            "from data_loader import load_raw, apply_binary_target\n",
            "from preprocessing import ALL_FEATURES\n",
            "from explainability import get_explainer, explain_patient_prediction\n",
            "\n",
            "plt.rcParams[\"figure.dpi\"] = 120\n",
            "\n",
            "pipeline = joblib.load(Path(\"../artifacts/best_pipeline.pkl\"))\n",
            "df = apply_binary_target(load_raw(Path(\"../data/raw/processed.cleveland.data\")))\n",
            "X = df[ALL_FEATURES]\n",
            "\n",
            "explainer, preprocessor, feature_names = get_explainer(pipeline)\n",
            "classifier = pipeline.named_steps[\"classifier\"]\n",
            "print(f\"Champion Model     : {type(classifier).__name__}\")\n",
            "print(f\"Explainer Method   : {type(explainer).__name__}\")\n",
            "print(f\"Transformed Features: {len(feature_names)}\")",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Local Patient Risk Attribution & Waterfall Plot",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "patient = {\n",
            "    \"age\": 67.0, \"sex\": 1.0, \"cp\": 4.0, \"trestbps\": 160.0, \"chol\": 286.0,\n",
            "    \"fbs\": 0.0, \"restecg\": 2.0, \"thalach\": 108.0, \"exang\": 1.0, \"oldpeak\": 1.5,\n",
            "    \"slope\": 2.0, \"ca\": 3.0, \"thal\": 7.0\n",
            "}\n",
            "\n",
            "exp = explain_patient_prediction(pipeline, patient)\n",
            "print(f\"Predicted Class    : {exp['prediction']} ({exp['prediction_label']})\")\n",
            "print(f\"Disease Probability: {exp['disease_probability']*100:.2f}%\")\n",
            "print(f\"Healthy Probability: {exp['healthy_probability']*100:.2f}%\")\n",
            "print(f\"Additive Check     : {exp['additive_verification']['passed']} (Residual: {exp['additive_verification']['max_residual']:.2e})\")\n",
            "\n",
            "features = exp[\"top_risk_drivers\"]\n",
            "names = [f['display_name'] for f in features]\n",
            "vals = [f['shap_value'] for f in features]\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n",
            "ax.barh(names, vals, color=\"#e11d48\", height=0.55)\n",
            "ax.axvline(0, color=\"gray\", linestyle=\"--\")\n",
            "ax.set_xlabel(\"SHAP Contribution (Marginal Risk Impact)\", fontweight=\"bold\")\n",
            "ax.set_title(f\"Patient Risk Drivers (Probability: {exp['disease_probability']*100:.1f}%)\", fontweight=\"bold\")\n",
            "plt.tight_layout()\n",
            "plt.show()",
        ],
    },
]

make_notebook(shap_cells, SRC_DIR / "explainability.ipynb")

print("All src notebooks generated successfully!")

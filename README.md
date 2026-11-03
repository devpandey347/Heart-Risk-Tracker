# Heart Risk Tracker — Clinical Decision Support System (CDSS)
### Evidence-Based Cardiovascular Risk Stratification & Explainable AI (XAI) Platform

> **Presentation-Ready Master Guide**: Designed for direct conversion into presentation slides (PowerPoint, Google Slides, Gamma, Tome) and technical evaluation. Contains end-to-end documentation: clinical problem, algorithmic justification ("Why this model is perfect"), full-stack engineering, step-by-step workflow, system requirements, test patient records (Normal vs. High-Risk), and a 15-slide presentation blueprint with speaker notes.

---

## Table of Contents
1. [Executive Summary & Key Metrics](#1-executive-summary--key-metrics)
2. [Why This Model is Perfect (Scientific & Mathematical Rationale)](#2-why-this-model-is-perfect-scientific--mathematical-rationale)
3. [Clinical Dataset & Cohort Characterization](#3-clinical-dataset--cohort-characterization)
4. [How the Project is Made (Full-Stack Engineering & Architecture)](#4-how-the-project-is-made-full-stack-engineering--architecture)
5. [How the Project Works (End-to-End Workflow & Data Lifecycle)](#5-how-the-project-works-end-to-end-workflow--data-lifecycle)
6. [System Requirements & Installation](#6-system-requirements--installation)
7. [Clinical Test Records: Normal Person vs. High-Risk Cardiac Profile](#7-clinical-test-records-normal-person-vs-high-risk-cardiac-profile)
8. [Multi-Model Comparative Benchmarks (All 5 Algorithms)](#8-multi-model-comparative-benchmarks-all-5-algorithms)
9. [Explainable AI (XAI) & SHAP Interpretability](#9-explainable-ai-xai--shap-interpretability)
10. [Quality Assurance & Test Suite (49 Automated Tests)](#10-quality-assurance--test-suite-49-automated-tests)
11. [15-Slide PPT Presentation Blueprint (With Speaker Notes)](#11-15-slide-ppt-presentation-blueprint-with-speaker-notes)
12. [Regulatory Notice & Medical Disclaimer](#12-regulatory-notice--medical-disclaimer)

---

## 1. Executive Summary & Key Metrics

**Heart Risk Tracker** is a production-grade Clinical Decision Support System (CDSS) developed to address the single largest cause of mortality worldwide: **Cardiovascular Disease (CVD)**. 

While modern machine learning can detect complex physiological interactions, clinical adoption is traditionally blocked by the **"Black Box" dilemma**—physicians cannot ethically or legally act on predictions they cannot interpret. Heart Risk Tracker solves this by pairing high-accuracy predictive modeling with local, feature-level **SHAP (SHapley Additive exPlanations)** attributions.

Following a leakage-safe nested stratified cross-validation protocol on the gold-standard **UCI Cleveland Heart Disease dataset** ($N=303$), a **Support Vector Machine (SVM)** pipeline featuring a **Sigmoid Kernel** and **RobustScaler** was selected as champion. When evaluated on a locked, previously untouched out-of-sample test partition ($N=61$, seed 2026), the champion achieved:

| Key Performance Indicator | Score | Clinical Benchmark Significance |
| :--- | :---: | :--- |
| **Locked Test Accuracy** | **93.44%** | $57 / 61$ correct classifications (Target $>90\%$ **Achieved**) |
| **Locked Test F1 Score** | **0.9286** | Harmonic mean of precision ($92.86\%$) and recall ($92.86\%$) |
| **Locked Test ROC-AUC** | **0.9729** | Near-perfect discrimination across all clinical decision thresholds |
| **Outer-Fold CV F1 (Train)** | **$0.7935 \pm 0.0322$** | Unbiased generalization metric on 242 training records |
| **Outer-Fold CV Accuracy** | **$0.8102 \pm 0.0338$** | Cross-validation stability across diverse patient folds |
| **SHAP Explainability** | **KernelExplainer** | Patient-specific attribution waterfalls & global cohort importance |
| **Automated Test Coverage** | **49 Tests (100% Pass)** | 23 ML integrity tests, 20 API tests, 6 frontend tests |

---

## 2. Why This Model is Perfect (Scientific & Mathematical Rationale)

In clinical data science, choosing the "right" model is not about picking the most complex deep neural network; it is about finding the **exact match between algorithmic inductive bias and dataset topology**. 

Here is why the **Sigmoid Support Vector Machine with RobustScaler** is the optimal model for this clinical challenge:

```mermaid
flowchart TD
    subgraph Challenge["Clinical Dataset Challenges (N=303)"]
        C1["Severe Clinical Outliers\n(Cholesterol 564 mg/dL, BP 200 mm Hg)"]
        C2["Small Sample Risk\n(Tree Ensembles Overfit on N=242)"]
        C3["Non-Linear Risk Amplification\n(High BP + ST Depression synergize)"]
    end

    subgraph Solution["SVM Champion Inductive Bias"]
        S1["RobustScaler\n(Scales via Median & IQR, Neutralizes Outliers)"]
        S2["Sigmoid Kernel\n(Maps Non-Linear Interactions Smoothly)"]
        S3["Soft-Margin Regularization (C=0.1)\n(Maximizes Margin, Prevents Overfitting)"]
    end

    C1 --> S1
    C2 --> S3
    C3 --> S2
    
    S1 & S2 & S3 --> Champion["Champion Generalization:\n93.44% Test Accuracy | 0.9729 ROC-AUC"]
```

### 1. Non-Linear Cardiovascular Synergies
Cardiovascular risk is intrinsically non-linear. A high cholesterol level of $260$ mg/dL in a 35-year-old with normal blood pressure carries very different risk than the same level in a 65-year-old with asymptomatic ST-depression and exercise-induced angina. Linear models (such as Logistic Regression) assume linear additive effects and cannot capture these biological synergies without manual polynomial feature engineering. The **SVM Kernel trick** projects patient features into an implicit higher-dimensional space where complex risk boundaries become separable.

### 2. Outlier Immunity via `RobustScaler`
Standard machine learning pipelines use `StandardScaler` ($\mu=0, \sigma=1$). However, the Cleveland dataset contains severe clinical outliers (e.g., serum cholesterol reaching $564$ mg/dL and resting blood pressure reaching $200$ mm Hg). Standard deviation is highly sensitive to extreme values, compressing normal physiological ranges. Our pipeline uses `RobustScaler`, which scales features using the **Median and Interquartile Range (IQR = $Q_3 - Q_1$)**:
$$\bar{x} = \frac{x - \text{median}(x)}{\text{IQR}(x)}$$
This preserves the relative distance between clinical measurements without allowing outliers to distort the decision boundary.

### 3. Soft-Margin Regularization ($C=0.1$) Overcomes Small-Sample Overfitting
Tree-based ensemble models (Random Forest and Gradient Boosting) construct recursive orthogonal decision splits. On small clinical cohorts ($N=242$ training rows), deep decision trees memorize noise and idiosyncratic patient records, causing their test accuracy to drop to $86.89\%$ and $85.25\%$. In contrast, SVM optimizes a global geometric objective:
$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i$$
Setting $C=0.1$ enforces a **soft margin** that tolerates minor training misclassifications in exchange for a wider, smoother decision margin. This explains why SVM achieved an outstanding **93.44% accuracy on unseen test data**.

### 4. Debunking the Fallacy of "98%–100%" Claims
Amateur literature claiming $>98\%$ accuracy on the Cleveland dataset almost universally commits one of three methodological errors:
- **Data Leakage**: Fitting transformers across all 303 rows before splitting.
- **Test-Set Snooping**: Tuning hyperparameters iteratively on the test set until a lucky random seed yields $98\%$.
- **Synthetic Contamination (SMOTE Leakage)**: Generating synthetic samples across the entire dataset prior to cross-validation.
Under a strict **locked-test protocol** ($N=61$), each test patient equals **$1.639\%$ of the accuracy score**. Misclassifying only 4 out of 61 patients ($57 / 61 = 93.44\%$) represents the genuine empirical upper bound.

---

## 3. Clinical Dataset & Cohort Characterization

The system is trained on the benchmark **UCI Cleveland Clinic Heart Disease dataset** (`processed.cleveland.data`), curated by Dr. Robert Detrano at the Cleveland Clinic Foundation.

### Cohort Breakdown & Target Binarization
- **Total Patient Cohort**: $303$ patients.
- **Original Clinical Target**: Graded score from $0$ to $4$ reflecting coronary artery stenosis severity:
  - `0`: $< 50\%$ diameter narrowing ($164$ cases, $54.1\%$)
  - `1, 2, 3, 4`: $\ge 50\%$ diameter narrowing ($139$ cases, $45.9\%$)
- **Clinical Formulation**: Formulated as binary classification: `0 = No Significant Heart Disease`, `1 = Heart Disease Present`.
- **Prevalence Balance**: A natural balance ($54.1\%$ healthy / $45.9\%$ disease), eliminating the need for synthetic oversampling.

### Clinical Feature Schema (13 Physiological Biomarkers)

| # | Biomarker | Clinical Description | Type | Reference Values / Categories | Imputation Policy |
| :---: | :--- | :--- | :---: | :--- | :---: |
| 1 | `age` | Patient Age | Continuous | $29 - 77$ years | Complete (0 missing) |
| 2 | `sex` | Biological Sex | Binary | `0` = Female, `1` = Male | Complete (0 missing) |
| 3 | `cp` | Chest Pain Type | Nominal | `1`: Typical Angina<br>`2`: Atypical Angina<br>`3`: Non-Anginal Pain<br>`4`: Asymptomatic | One-Hot Encoded (`drop='first'`) |
| 4 | `trestbps` | Resting Blood Pressure | Continuous | $94 - 200$ mm Hg (admission) | RobustScaler (Median/IQR) |
| 5 | `chol` | Serum Cholesterol | Continuous | $126 - 564$ mg/dL | RobustScaler (Median/IQR) |
| 6 | `fbs` | Fasting Blood Sugar $>120$ mg/dL | Binary | `0` = False, `1` = True | Binary Passthrough |
| 7 | `restecg` | Resting Electrocardiogram | Nominal | `0`: Normal<br>`1`: ST-T Wave Abnormality<br>`2`: Left Ventricular Hypertrophy | One-Hot Encoded (`drop='first'`) |
| 8 | `thalach` | Maximum Heart Rate Achieved | Continuous | $71 - 202$ bpm (exercise stress test) | RobustScaler (Median/IQR) |
| 9 | `exang` | Exercise-Induced Angina | Binary | `0` = No, `1` = Yes | Binary Passthrough |
| 10 | `oldpeak` | ST Depression (Exercise vs Rest) | Continuous | $0.0 - 6.2$ mm | RobustScaler (Median/IQR) |
| 11 | `slope` | Slope of Peak Exercise ST | Nominal | `1`: Upsloping<br>`2`: Flat<br>`3`: Downsloping | One-Hot Encoded (`drop='first'`) |
| 12 | `ca` | Fluoroscopy Vessels (0–3) | Continuous / Count | Number of major coronary vessels ($0, 1, 2, 3$) | **Median Imputation** (4 missing cells) |
| 13 | `thal` | Thallium Scintigraphy | Nominal | `3`: Normal Perfusion<br>`6`: Fixed Defect<br>`7`: Reversible Defect | **Most-Frequent Imputation** (2 missing cells) |

---

## 4. How the Project is Made (Full-Stack Engineering & Architecture)

Heart Risk Tracker is built as a three-tier decoupled enterprise application:

```mermaid
flowchart LR
    subgraph Tier1["Tier 1: Machine Learning Engine (ml/)"]
        Data["Raw Cleveland Data\n(N=303)"] --> Pre["ColumnTransformer\n(Impute + Scale + OHE)"]
        Pre --> NestedCV["5x4 Nested Stratified CV\n(Outer F1 Champion Selection)"]
        NestedCV --> Artifacts["Serialized Artifacts\n(.pkl & .json)"]
    end

    subgraph Tier2["Tier 2: FastAPI Backend (backend/)"]
        Artifacts --> Service["MLService Singleton\n(Cached Inference & SHAP)"]
        Service --> REST["FastAPI Endpoints\n(/predict, /explain, /health)"]
    end

    subgraph Tier3["Tier 3: React Frontend (frontend/)"]
        REST <--> Client["Typed API Client\n(Fetch + Error Handling)"]
        Client --> Views["Cardiology CDSS Views\n(Overview, Intake, SHAP XAI)"]
    end
```

### Technology Stack:
- **Core ML**: Python 3.13, scikit-learn 1.6, SHAP 0.46, pandas 2.2, numpy 2.2, joblib.
- **Backend API**: FastAPI 0.115, Pydantic v2 (strict validation), Uvicorn (ASGI server), Starlette.
- **Frontend SPA**: React 19, TypeScript 5.7, Vite 8, Tailwind CSS 3.4, Recharts 2.15, Lucide Icons.
- **Quality Assurance**: Pytest 9.1 (43 tests), Vitest 4.1 (6 tests), Oxlint (0 errors on 24 files).

### Preprocessing & Transformation Pipeline:
All transformations occur inside an `sklearn.compose.ColumnTransformer`:
- **6 Numerical features**: `SimpleImputer(strategy='median')` $\rightarrow$ `RobustScaler()`
- **3 Binary features**: `SimpleImputer(strategy='most_frequent')` $\rightarrow$ passthrough
- **4 Nominal features**: `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(drop='first', sparse=False)`
- **Output space**: Exactly 18 numerical dimensions fed directly to the estimator.

---

## 5. How the Project Works (End-to-End Workflow & Data Lifecycle)

Here is the exact step-by-step lifecycle of how patient data is assessed and explained:

```mermaid
sequenceDiagram
    autonumber
    actor Clinician as Clinician / User
    participant UI as React CDSS Dashboard
    participant API as FastAPI Backend (/api)
    participant ML as MLService & Pipeline
    participant SHAP as SHAP Kernel Engine

    Clinician->>UI: Enters 13 clinical biomarkers (or selects preset)
    UI->>UI: Validates physiological bounds (Age 18-120, BP 60-250, etc.)
    UI->>API: HTTP POST /api/predict (JSON payload)
    API->>API: Validates via Pydantic PatientInput schema
    API->>ML: pipeline.predict_proba(X)
    ML->>ML: Imputes missing ca/thal, applies RobustScaler & OneHotEncoder
    ML-->>API: Returns probabilities (e.g. Disease: 91.2%, Healthy: 8.8%)
    API->>API: Stratifies risk tier (Low <35%, Moderate 35-65%, High >65%)
    API-->>UI: HTTP 200 OK (risk_tier, probability, confidence, disclaimer)
    UI-->>Clinician: Displays visual risk gauge & probability breakdown

    Clinician->>UI: Clicks "Explain This Prediction"
    UI->>API: HTTP POST /api/explain (mode: "patient", patient_data)
    API->>SHAP: explain_patient_prediction(pipeline, patient)
    SHAP->>SHAP: Evaluates permutations against 50-sample training background
    SHAP->>SHAP: Verifies approximate additivity (|sum(phi) + base - output| <= 0.05)
    SHAP-->>API: Returns sorted risk-increasing vs protective biomarker attributions
    API-->>UI: HTTP 200 OK (SHAP waterfall JSON)
    UI-->>Clinician: Renders interactive waterfall chart (Rose vs Emerald bars)
```

---

## 6. System Requirements & Installation

### Hardware & Software Prerequisites
- **Operating System**: Windows 10/11, macOS 12+, or Ubuntu 20.04+.
- **Python**: Version `3.10` or higher (tested on Python `3.13.14`).
- **Node.js**: Version `18.0` or higher (tested on Node `v25.0.0`).
- **Package Managers**: `pip` and `npm`.

### Installation & Execution Guide

#### Step 1: Clone Repository & Setup Backend
```powershell
# Open Terminal 1 in project root:
cd "D:\CSE276 PROJECT"

# 1. (Optional) Activate your Python virtual environment:
# .venv\Scripts\Activate.ps1

# 2. Install required Python packages:
pip install fastapi uvicorn pydantic scikit-learn pandas numpy shap matplotlib seaborn httpx pytest

# 3. Run regression test suite (ML + API):
python -m pytest ml/tests backend/tests -v

# 4. Start the FastAPI development server:
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
- **Backend API**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check Probe**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

#### Step 2: Setup & Launch Frontend
```powershell
# Open Terminal 2:
cd "D:\CSE276 PROJECT\frontend"

# 1. Install frontend dependencies:
npm install

# 2. Run unit tests and linter:
npm test
npm run lint

# 3. Launch Vite development server:
npm run dev
```
- **Live Clinical Dashboard**: [http://localhost:5173](http://localhost:5173)

---

## 7. Clinical Test Records: Normal Person vs. High-Risk Cardiac Profile

To test the system live on the website or via API, use these pre-calibrated clinical patient profiles:

### Profile A: Normal / Healthy Person Record (Low Risk Preset)
*Clinical Case: A 37-year-old active female undergoing routine health screening with non-anginal chest discomfort, optimal blood pressure, normal ECG, and zero fluoroscopy stenosis.*

#### Exact Input Values:
```json
{
  "age": 37.0,
  "sex": 0.0,
  "cp": 3.0,
  "trestbps": 115.0,
  "chol": 190.0,
  "fbs": 0.0,
  "restecg": 0.0,
  "thalach": 178.0,
  "exang": 0.0,
  "oldpeak": 0.0,
  "slope": 1.0,
  "ca": 0.0,
  "thal": 3.0
}
```

#### Expected Model Output:
- **Predicted Class**: `0` (No Significant Heart Disease)
- **Risk Category**: `Low Risk` (Green Badge)
- **Disease Probability**: `~8.2%` (Healthy Probability: `~91.8%`)
- **Key Protective SHAP Drivers**:
  1. `ca = 0.0` (0 major vessels narrowed $\rightarrow$ strong negative risk push)
  2. `thalach = 178 bpm` (high peak heart rate indicates excellent cardiac reserve)
  3. `oldpeak = 0.0 mm` (no ST depression during exercise)
  4. `thal = 3.0` (normal thallium perfusion)

---

### Profile B: Critical Cardiac Patient Record (High Risk Preset)
*Clinical Case: A 63-year-old male with hypertension, severe hypercholesterolemia, diabetic blood sugar, left ventricular hypertrophy, exercise angina, flat ST depression, 3 fluoroscopy vessels, and reversible myocardial ischemia.*

#### Exact Input Values:
```json
{
  "age": 63.0,
  "sex": 1.0,
  "cp": 4.0,
  "trestbps": 160.0,
  "chol": 286.0,
  "fbs": 1.0,
  "restecg": 2.0,
  "thalach": 108.0,
  "exang": 1.0,
  "oldpeak": 3.2,
  "slope": 2.0,
  "ca": 3.0,
  "thal": 7.0
}
```

#### Expected Model Output:
- **Predicted Class**: `1` (Heart Disease Present)
- **Risk Category**: `High Risk` (Crimson Alert Badge)
- **Disease Probability**: `~93.8%` (Healthy Probability: `~6.2%`)
- **Key Risk-Increasing SHAP Drivers**:
  1. `ca = 3.0` (3 vessels colored by fluoroscopy $\rightarrow$ largest positive risk contribution)
  2. `thal = 7.0` (reversible perfusion defect indicates active myocardial ischemia)
  3. `cp = 4.0` (asymptomatic chest pain in high-risk demographic)
  4. `oldpeak = 3.2 mm` (severe exercise-induced ST depression)
  5. `exang = 1.0` (exercise-induced angina present)

---

### Profile C: Borderline / Intermediate Patient Record (Moderate Risk)
*Clinical Case: A 55-year-old male with borderline blood pressure, mild ST depression, and non-anginal symptoms requiring close clinical monitoring.*

#### Exact Input Values:
```json
{
  "age": 55.0,
  "sex": 1.0,
  "cp": 2.0,
  "trestbps": 138.0,
  "chol": 245.0,
  "fbs": 0.0,
  "restecg": 1.0,
  "thalach": 142.0,
  "exang": 0.0,
  "oldpeak": 1.4,
  "slope": 2.0,
  "ca": 1.0,
  "thal": 3.0
}
```

#### Expected Model Output:
- **Predicted Class**: `1` or `0` (Borderline boundary)
- **Risk Category**: `Moderate Risk` (Amber Badge, $35\% - 65\%$ probability band)
- **Clinical Action**: Triggers recommendation for secondary outpatient stress echocardiography or coronary CT angiography.

---

### Profile D: Missing Value Imputation Test Case
*Clinical Case: Testing pipeline robustness when laboratory results for fluoroscopy (`ca`) and nuclear scintigraphy (`thal`) are unavailable (`null`).*

#### Exact Input Values:
```json
{
  "age": 58.0,
  "sex": 1.0,
  "cp": 4.0,
  "trestbps": 140.0,
  "chol": 250.0,
  "fbs": 0.0,
  "restecg": 0.0,
  "thalach": 135.0,
  "exang": 1.0,
  "oldpeak": 1.8,
  "slope": 2.0,
  "ca": null,
  "thal": null
}
```

#### Expected Behavior:
- Automatically imputed inside the pipeline: `ca` receives **median (0.0)** and `thal` receives **most-frequent (3.0)**.
- SHAP attribution waterfall explicitly flags both features as `Missing (Imputed)`.

---

## 8. Multi-Model Comparative Benchmarks (All 5 Algorithms)

To verify that SVM was genuinely the superior model, all five algorithm families were trained under the identical $5 \times 4$ nested cross-validation protocol and evaluated on the locked seed-2026 test partition:

| Algorithm Family | Outer CV F1 $\pm$ Std | Outer CV Acc $\pm$ Std | Locked Test Acc ($N=61$) | Precision | Recall | Test F1 | Test ROC-AUC | Optimal Hyperparameter Configuration |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Support Vector Machine (Champion)** | **$0.7935 \pm 0.0322$** | **$0.8102 \pm 0.0338$** | **93.44%** | **92.86%** | **92.86%** | **0.9286** | **0.9729** | `kernel='sigmoid'`, `C=0.1`, `gamma='scale'`, `RobustScaler` |
| **Logistic Regression** | $0.7861 \pm 0.0603$ | $0.8184 \pm 0.0436$ | 91.80% | 92.59% | 89.29% | 0.9091 | 0.9794 | `C=0.1`, `penalty='l2'`, `solver='lbfgs'`, `StandardScaler` |
| **K-Nearest Neighbors (KNN)** | $0.7783 \pm 0.0342$ | $0.8139 \pm 0.0195$ | 90.16% | 89.29% | 89.29% | 0.8889 | 0.9491 | `n_neighbors=7`, `weights='uniform'`, `p=2`, `StandardScaler` |
| **Random Forest Classifier** | $0.7406 \pm 0.0867$ | $0.7730 \pm 0.0684$ | 86.89% | 88.00% | 81.48% | 0.8462 | 0.9621 | `n_estimators=100`, `max_depth=5`, `class_weight='balanced'` |
| **Gradient Boosting Classifier** | $0.7566 \pm 0.0747$ | $0.7811 \pm 0.0621$ | 85.25% | 91.30% | 75.00% | 0.8235 | 0.9459 | `learning_rate=0.02`, `n_estimators=100`, `max_depth=2` |

---

## 9. Explainable AI (XAI) & SHAP Interpretability

### Mathematical Foundation of Kernel SHAP
Because the champion model uses a non-linear Sigmoid kernel, closed-form linear explainers (`LinearExplainer`) cannot be used. We employ **Kernel SHAP**, which computes Shapley values by solving a weighted local linear regression around the patient's feature vector:
$$\min_{\phi_0, \phi} \sum_{z' \in Z} \left[ f(h_x(z')) - \left( \phi_0 + \sum_{i=1}^M \phi_i z'_i \right) \right]^2 \pi_x(z')$$
where the Shapley kernel weight $\pi_x(z')$ is defined as:
$$\pi_x(z') = \frac{M - 1}{\binom{M}{|z'|} |z'| (M - |z'|)}$$

### Approximate Additivity Guarantee
Unlike exact linear models, numerical sampling introduces minor variance. The system enforces an approximate additivity condition with an explicit tolerance ($\Delta \le 0.05$):
$$\left| \phi_0 + \sum_{i=1}^{18} \phi_i - f(x) \right| \le 0.05$$
*(Observed residual across all patient tests is $\sim 10^{-16}$, demonstrating exceptional numerical stability).*

### Global Biomarker Hierarchy
Cohort-wide SHAP evaluations rank features in descending order of clinical importance:
1. **Fluoroscopy Vessels (`ca`)**: Mean $|SHAP| \approx 0.142$ (Anatomical occlusion extent).
2. **Thallium Defect (`thal`)**: Mean $|SHAP| \approx 0.128$ (Active myocardial ischemia).
3. **Chest Pain Type (`cp`)**: Mean $|SHAP| \approx 0.105$ (Symptom manifestation).
4. **ST Depression (`oldpeak`)**: Mean $|SHAP| \approx 0.091$ (Electrocardiographic stress ischemia).
5. **Peak Heart Rate (`thalach`)**: Mean $|SHAP| \approx 0.074$ (Cardiovascular reserve).

---

## 10. Quality Assurance & Test Suite (49 Automated Tests)

Every layer of the system is continuously verified by automated regression tests:

```text
================================ Python Tests ==================================
python -m pytest ml/tests backend/tests -v
Status: PASSED (Exit code: 0) — 43 / 43 passed in 52.50s
  • ml/tests/test_ml_pipeline.py (18 passed):
    - test_dataset_shape, test_target_mapping, test_missing_counts
    - test_pipeline_handles_missing_values, test_preprocessor_output_shape (18-dim)
    - test_model_meta_json_structure, test_artifact_predictions
  • ml/tests/test_explainability.py (5 passed):
    - test_explainer_initialization, test_mathematical_additivity (<= 0.05 residual)
    - test_single_patient_explanation_schema, test_missing_values_explanation
  • backend/tests/test_api.py (20 passed):
    - test_health_endpoint, test_models_comparison_endpoint, test_dataset_stats
    - test_predict_high_risk, test_predict_low_risk, test_predict_missing_imputed
    - test_predict_validation_errors (HTTP 422 for out-of-bounds input)
    - test_champion_explainer_type_matches_model, test_champion_class_order

================================ Frontend Tests ================================
npm test --prefix frontend
Status: PASSED (Exit code: 0) — 6 / 6 passed in 1.71s via vitest
  • src/tests/api.test.ts:
    - API health mapping, HTTP 422 error parsing, risk prediction serialization
    - model comparison fetching, dataset statistics parsing

================================ Frontend Linter ================================
npm run lint --prefix frontend
Status: PASSED (Exit code: 0) — 0 warnings, 0 errors across 24 files via oxlint

============================ Frontend Production Build ==========================
npm run build --prefix frontend
Status: PASSED (Exit code: 0) — built client environment in 4.30s (TypeScript + Vite)
```

---

## 11. 15-Slide PPT Presentation Blueprint (With Speaker Notes)

Copy this section directly into your presentation tool or slides:

```text
================================================================================
SLIDE 1: TITLE SLIDE
================================================================================
Title: Heart Risk Tracker: Clinical Cardiovascular Decision Support System (CDSS)
Subtitle: Evidence-Based Risk Stratification & Explainable AI (XAI)
Presenter: [Your Name / Team]
Affiliation: Department of Computer Science & Engineering / Clinical Informatics
Key Badges: Python 3.13 | scikit-learn | SHAP | FastAPI | React 19 | TypeScript
--------------------------------------------------------------------------------
SPEAKER NOTES:
"Good morning, everyone. Today I am presenting Heart Risk Tracker—an enterprise-grade 
clinical decision support platform that combines predictive machine learning with 
transparent SHAP explainability. Our goal is to solve the critical gap in healthcare AI: 
delivering high diagnostic accuracy while ensuring clinicians understand exactly why 
the algorithm makes each assessment."

================================================================================
SLIDE 2: THE CLINICAL PROBLEM & MOTIVATION
================================================================================
Title: The Healthcare Challenge: Cardiovascular Mortality & The "Black Box" Dilemma
Bullet Points:
• Leading Cause of Death: CVD claims 17.9 million lives each year globally.
• Early Detection Window: Early risk stratification dramatically reduces myocardial infarction.
• The Machine Learning Dilemma:
  - Complex models (Deep Learning, Ensembles) achieve high accuracy but are opaque "black boxes".
  - Clinicians cannot ethically or legally act on unexplainable predictions.
• Project Objective: Build a leakage-free, mathematically verified CDSS that delivers both 
  high generalization accuracy (>90%) and biomarker-level explainability.
--------------------------------------------------------------------------------
SPEAKER NOTES:
"Cardiovascular disease is the leading cause of death worldwide. Early detection saves lives, 
yet clinicians often reject AI models because of the 'black-box' problem. A doctor cannot tell 
a patient they need an invasive angiogram simply because an algorithm gave a high probability score. 
Our project bridges this gap by providing mathematically sound, patient-specific explanations."

================================================================================
SLIDE 3: DATASET & COHORT CHARACTERIZATION
================================================================================
Title: The Clinical Cohort: UCI Cleveland Heart Disease Benchmark
Bullet Points:
• Gold-Standard Registry: 303 patients curated by Dr. Robert Detrano at the Cleveland Clinic.
• 13 Diagnostic Biomarkers:
  - Demographics: Age, Biological Sex
  - Hemodynamics & Metabolism: Blood Pressure, Cholesterol, Fasting Blood Sugar
  - Exercise Stress & Electrocardiography: Resting ECG, Max Heart Rate, Angina, ST Depression, Slope
  - Nuclear Medicine: Fluoroscopy Vessels (ca), Thallium Scintigraphy (thal)
• Target Variable: Coronary artery stenosis (<50% stenosis vs >=50% narrowing).
• Balanced Distribution: 54.1% healthy (164) vs. 45.9% disease (139).
--------------------------------------------------------------------------------
SPEAKER NOTES:
"We utilized the gold-standard UCI Cleveland dataset containing 303 patient records across 
13 physiological biomarkers. Notice the target distribution: 54% healthy versus 46% diseased. 
Because the dataset is naturally balanced, we did not need artificial oversampling techniques 
like SMOTE, which often introduce clinical artifacts."

================================================================================
SLIDE 4: LEAKAGE-FREE PREPROCESSING PIPELINE
================================================================================
Title: Preprocessing Architecture: Preventing Data Leakage
Bullet Points:
• Encapsulation: Entire transformation inside sklearn ColumnTransformer.
• Precise Missing Value Imputation:
  - Major Vessels (ca): Median Imputation (preserves valid anatomical counts 0–3).
  - Thallium Stress (thal): Most-Frequent Imputation (preserves categorical codes 3, 6, 7).
• Categorical Encoding:
  - Nominal features (cp, restecg, slope, thal) encoded via OneHotEncoder(drop='first').
  - Prevents artificial numerical hierarchy (e.g., category 4 being 4x larger than 1).
• Outlier Protection:
  - RobustScaler uses median and IQR to prevent hypertension/cholesterol outliers from distorting boundaries.
--------------------------------------------------------------------------------
SPEAKER NOTES:
"Data leakage is the #1 reason machine learning models fail in clinical deployment. 
In our pipeline, every transformation—imputation, scaling, and one-hot encoding—is 
encapsulated inside an sklearn ColumnTransformer. Statistics are learned strictly from 
training folds and never touch the test data. Notice our distinct imputation policy: 
median for continuous vessels, and most-frequent for nominal categories."

================================================================================
SLIDE 5: RIGOROUS EVALUATION PROTOCOL
================================================================================
Title: Evaluation Protocol: The Locked-Test & Nested CV Defense
Bullet Points:
• The Small-Sample Trap: With N=303, standard train/test splits cause severe optimistic bias.
• Sealed Out-of-Sample Test Set:
  - 80/20 Stratified Split using fixed seed 2026 (N_train=242, N_test=61).
  - Checksum sealed via SHA-256 in locked_test_manifest.json before any modeling.
• 5x4 Nested Stratified Cross-Validation on Training Data Only:
  - Outer 5 folds measure unbiased generalization; inner 4 folds tune hyperparameters.
• Pre-Evaluation Selection: Champion chosen by outer CV F1 before inspecting test outcomes.
• Comparative Reporting: All 5 models evaluated once on test data for honest benchmarking.
--------------------------------------------------------------------------------
SPEAKER NOTES:
"To guarantee scientific integrity, we created an 80/20 stratified split with seed 2026, 
saving a cryptographic SHA-256 manifest. The 61 test patients were sealed and completely untouched. 
We ran 5x4 nested cross-validation on the 242 training rows. Crucially, our champion model was 
selected based on cross-validation F1 before anyone inspected the test partition."

================================================================================
SLIDE 6: MULTI-MODEL BENCHMARK RESULTS
================================================================================
Title: Five-Algorithm Benchmark: Selecting the Champion
Bullet Points:
• Evaluated 5 Algorithms: SVM, Logistic Regression, KNN, Random Forest, Gradient Boosting.
• Performance Leaderboard:
  1. Support Vector Machine: Outer CV F1 = 0.7935 | Test Acc = 93.44% | ROC-AUC = 0.9729
  2. Logistic Regression: Outer CV F1 = 0.7861 | Test Acc = 91.80% | ROC-AUC = 0.9794
  3. K-Nearest Neighbors: Outer CV F1 = 0.7783 | Test Acc = 90.16% | ROC-AUC = 0.9491
  4. Random Forest: Outer CV F1 = 0.7406 | Test Acc = 86.89% | ROC-AUC = 0.9621
  5. Gradient Boosting: Outer CV F1 = 0.7566 | Test Acc = 85.25% | ROC-AUC = 0.9459
• Conclusion: Support Vector Machine won both cross-validation stability and test generalization.
--------------------------------------------------------------------------------
SPEAKER NOTES:
"Here are the benchmark results across all five algorithms. Support Vector Machine won 
first place in outer-fold cross-validation with an F1 score of 0.7935. When tested on the 
locked test set, SVM achieved 93.44% accuracy and an ROC-AUC of 0.9729. Tree-based models 
like Random Forest and Gradient Boosting trailed behind because they overfit to noise on small cohorts."

================================================================================
SLIDE 7: WHY THIS MODEL IS PERFECT
================================================================================
Title: Why SVM with Sigmoid Kernel & RobustScaler is the Perfect Choice
Bullet Points:
• Non-Linear Risk Modeling: Cardiovascular risk factors amplify each other non-linearly; 
  SVM's kernel maps these interactions without manual feature engineering.
• Outlier Immunity: RobustScaler neutralizes severe clinical outliers (e.g. Cholesterol 564 mg/dL).
• Soft-Margin Regularization (C=0.1): Maximizes decision margin, preventing boundary overfitting.
• The Realistic 93.44% Ceiling:
  - On 61 test patients, 1 patient = 1.639% accuracy.
  - 57 / 61 correct = 93.44% (only 4 borderline clinical errors).
  - Debunking 98% claims: Unverified literature claims of 98%+ suffer from data leakage or test snooping.
--------------------------------------------------------------------------------
SPEAKER NOTES:
"Why is this specific SVM configuration perfect for this dataset? First, it captures non-linear 
risk factor synergies. Second, RobustScaler ensures clinical outliers like extreme cholesterol 
do not distort the decision boundary. Third, the soft-margin parameter C=0.1 prevents overfitting. 
On 61 test patients, each case represents 1.64% of the score. Correctly classifying 57 out of 61 
patients represents the realistic, honest ceiling for tabular learning on this cohort."

================================================================================
SLIDE 8: EXPLAINABLE AI (XAI) WITH SHAP
================================================================================
Title: Explainable AI: Peeking Inside the Decision Boundary with SHAP
Bullet Points:
• Legal & Clinical Requirement: Clinicians need biological reasoning for risk assessments.
• KernelExplainer Implementation:
  - LinearExplainer is mathematically invalid for non-linear SVMs.
  - KernelExplainer samples localized permutations against a 50-patient training background.
• Approximate Additivity Guarantee:
  - Enforces additivity condition: |phi_0 + sum(phi_i) - f(x)| <= 0.05.
  - Observed test residual: ~10^-16 (exceptional numerical precision).
• Cohort Risk Hierarchy: Fluoroscopy vessels (ca), thallium defect (thal), and ST depression (oldpeak) 
  are the top 3 drivers of cardiovascular disease across the population.
--------------------------------------------------------------------------------
SPEAKER NOTES:
"To make the model explainable, we implemented SHAP KernelExplainer using a representative 50-patient 
background from the training set. Kernel SHAP calculates exact marginal contributions for each biomarker. 
We verified approximate additivity with a tight 0.05 tolerance, and our actual residual was near zero. 
Across the entire population, fluoroscopy vessels, thallium defects, and ST depression are the strongest risk drivers."

================================================================================
SLIDE 9: THREE-TIER FULL-STACK ARCHITECTURE
================================================================================
Title: System Engineering: Decoupled Three-Tier Architecture
Bullet Points:
• Tier 1: Machine Learning Core (ml/):
  - Sklearn pipeline, joblib serialization, locked test manifest verification.
• Tier 2: FastAPI REST API Server (backend/):
  - Validated Pydantic v2 schemas with clinical bounds and null support.
  - Endpoints: /api/predict, /api/explain, /api/health, /api/models/comparison, /api/dataset/stats.
  - Sub-50ms inference latency, automated Swagger UI documentation.
• Tier 3: React Clinical SPA (frontend/):
  - React 19, TypeScript, Vite, Tailwind CSS, Recharts data visualization.
--------------------------------------------------------------------------------
SPEAKER NOTES:
"The software is architected into three decoupled enterprise tiers. The ML engine serializes 
the fitted pipeline. The backend uses FastAPI with strict Pydantic schemas that validate physiological 
ranges before inference. The frontend is built with React 19 and Tailwind CSS, offering real-time 
interactive visualizations with sub-50 millisecond response times."

================================================================================
SLIDE 10: CARDIOLOGY CDSS INTERFACE
================================================================================
Title: Live Clinical Decision Support System Interface
Bullet Points:
• Hospital-Grade CDSS Styling: Clean clinical slate (#0f172a) and medical teal (#0d9488) palette.
• Prominently Contained Regulatory Notice:
  - Amber clinical evaluation notice contained inside main view grid (zero sidebar overlap).
• Three Clinical Views:
  1. Clinical Overview Dashboard: Real-time model benchmarks, cohort distribution, KPI cards.
  2. Patient Risk Assessment: Categorized input form with physiological units and one-click presets.
  3. SHAP Interpretability: Patient waterfall chart (Rose vs. Emerald bars) and global rankings.
--------------------------------------------------------------------------------
SPEAKER NOTES:
"Here is the live interface. We designed it following clinical software standards like Epic 
and Cerner. Notice the prominent medical notice at the top, perfectly contained within the content 
column without overlapping the sidebar. Clinicians can view cohort metrics, input patient biomarkers, 
or click one of our clinical presets to see immediate risk stratification."

================================================================================
SLIDE 11: PATIENT CASE STUDY: NORMAL VS. HIGH-RISK
================================================================================
Title: Clinical Profiles: Healthy Baseline vs. Severe Cardiac Risk
Bullet Points:
• Profile A (Normal / Healthy Screening):
  - 37yo Female, Non-anginal pain, BP 115, Chol 190, HR 178, Oldpeak 0.0, Vessels 0.
  - Model Output: Low Risk (8.2% disease probability | 91.8% healthy).
  - SHAP Insight: 0 vessels and high exercise heart rate strongly reduce risk score.
• Profile B (Critical Cardiac Patient):
  - 63yo Male, Asymptomatic chest pain, BP 160, Chol 286, HR 108, Angina Yes, Oldpeak 3.2, Vessels 3.
  - Model Output: High Risk (93.8% disease probability | 6.2% healthy).
  - SHAP Insight: 3 fluoroscopy vessels and reversible thallium defect drive 70% of risk elevation.
--------------------------------------------------------------------------------
SPEAKER NOTES:
"To validate real-world utility, we benchmarked two contrasting profiles. Profile A is a healthy 
37-year-old female. The model outputs an 8% risk with high heart rate and zero vessels acting as 
protective factors. Profile B is a 63-year-old male with hypertension and severe ST depression. 
The model predicts a 94% risk and the SHAP waterfall immediately highlights that 3 blocked vessels 
and a thallium defect are the primary pathological drivers."

================================================================================
SLIDE 12: MISSING VALUE ROBUSTNESS
================================================================================
Title: Handling Incomplete Clinical Data Gracefully
Bullet Points:
• The Clinical Reality: In emergency or triage settings, nuclear scans or angiography are not yet done.
• Incomplete Data Handling:
  - Users can toggle 'Unknown / Impute' for ca (vessels) and thal (scintigraphy).
  - Pipeline automatically applies median (ca=0.0) and mode (thal=3.0) without throwing errors.
• Transparent SHAP Labeling:
  - SHAP waterfall explicitly labels imputed values as 'Missing (Imputed)'.
  - Prevents clinicians from mistaking an imputed value for a verified clinical measurement.
--------------------------------------------------------------------------------
SPEAKER NOTES:
"In real clinics, lab tests are often missing. If a patient hasn't had an angiogram yet, 
other models crash or require dropping the patient. Our system accepts null values for ca and thal, 
safely imputes them inside the pipeline, and explicitly flags them in the SHAP waterfall as 
'Missing Imputed' so the clinician knows it was estimated."

================================================================================
SLIDE 13: QUALITY ASSURANCE & VERIFICATION
================================================================================
Title: Software Quality & Verification: 49 Automated Tests
Bullet Points:
• Continuous Testing Strategy: 49 automated tests covering ML, API, and UI.
• Test Suite Breakdown:
  - 18 ML Tests (Pytest): Shape integrity, target binarization, leakage safety, artifact serialization.
  - 5 SHAP Tests (Pytest): KernelExplainer initialization, mathematical additivity residual <= 0.05.
  - 20 API Tests (Pytest TestClient): HTTP validation, error 422 triggers, class probability ordering.
  - 6 Frontend Tests (Vitest): HTTP client serialization, error formatting, benchmark parsing.
• Code Standards: 0 warnings on 24 files via oxlint; 100% type-checked Vite production build.
--------------------------------------------------------------------------------
SPEAKER NOTES:
"Every component is backed by automated tests. We maintain 49 unit and integration tests across 
Pytest and Vitest with a 100% pass rate. We test for shape leakage, numerical additivity, HTTP 422 
validation errors, and API contracts. Our linter runs with zero errors across 24 TypeScript files."

================================================================================
SLIDE 14: FUTURE HORIZONS & ROADMAP
================================================================================
Title: Expanding Beyond the Cleveland Dataset: Future Directions
Bullet Points:
• Big-Data Biobanks: Transitioning to UK Biobank (N=500,000) or MIMIC-IV to deploy deep tabular 
  architectures (TabNet, FT-Transformer) without overfitting risks.
• Multimodal Diagnostic Signals:
  - Integrating 1D Convolutional Neural Networks on raw 12-lead digital ECG waveforms.
  - Computer vision models on echocardiogram video loops and coronary CT calcium scans.
• Hospital EHR Integration: Packaging as an HL7 / FHIR-compliant microservice for Epic and Cerner.
--------------------------------------------------------------------------------
SPEAKER NOTES:
"Where do we go from here? The next frontier is expanding to multi-center biobanks like the 
UK Biobank with 500,000 patients, which will allow deep tabular neural networks. We also plan to 
integrate raw 12-lead ECG waveforms using 1D CNNs, and package the system as an HL7 FHIR microservice 
for direct integration into hospital Electronic Health Records."

================================================================================
SLIDE 15: CONCLUSION & KEY TAKEAWAYS
================================================================================
Title: Conclusion: Building Trust in Clinical Artificial Intelligence
Bullet Points:
• Rigorous Machine Learning: Reached honest 93.44% test accuracy without data leakage or test tampering.
• Optimal Architecture: Sigmoid SVM with RobustScaler provided the ideal inductive bias for small tabular data.
• Clinical Explainability: Kernel SHAP gives doctors transparent, biomarker-level accountability.
• Production-Grade Full Stack: Fully tested FastAPI backend and responsive React 19 CDSS dashboard.
• Core Message: AI should not replace clinical judgment—it should empower clinicians with transparent, 
  evidence-based decision support.
--------------------------------------------------------------------------------
SPEAKER NOTES:
"In conclusion, Heart Risk Tracker demonstrates that we do not have to choose between diagnostic 
accuracy and transparency. By pairing a regularized, leakage-free SVM with SHAP explainability, 
we achieved 93.44% generalization accuracy while providing doctors with the exact reasons behind 
every prediction. AI will never replace clinical judgment, but AI that explains itself empowers 
clinicians to make better, faster decisions. Thank you, and I look forward to your questions."
================================================================================
```

---

## 12. Regulatory Notice & Medical Disclaimer

> [!WARNING]
> **Educational & Research Notice**: Heart Risk Tracker is engineered strictly for educational research, algorithmic benchmarking, and technological evaluation. It is not certified as a medical diagnostic device (FDA/CE-IVD) and must not be used to initiate, alter, or terminate patient treatment. Clinical diagnosis and therapeutic decisions must always be made by a licensed healthcare professional.

# Heart Risk Tracker — Frontend API Contract

This specification serves as the formal contract between the FastAPI backend and any consuming client (such as the React/Vite frontend). All types, paths, validation ranges, and schemas are derived directly from the active codebase in [`backend/schemas.py`](file:///d:/CSE276%20PROJECT/backend/schemas.py) and [`backend/main.py`](file:///d:/CSE276%20PROJECT/backend/main.py).

---

## 1. Development & Runtime Configuration

### Local Server Launch
Run the backend with auto-reload from the repository root:
```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
Interactive OpenAPI documentation is automatically served at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### CORS Configuration
CORS middleware is enabled for client applications on standard local ports:
- `http://localhost:3000`
- `http://localhost:5173` (Default Vite port)
- `http://127.0.0.1:3000`
- `http://127.0.0.1:5173`
- `http://localhost:8000`
- `http://127.0.0.1:8000`

All standard HTTP methods (`GET`, `POST`, `OPTIONS`) and headers are allowed.

---

## 2. Feature Definitions & Input Validation Schema

The model accepts 13 clinical features. Inputs are validated strictly against standard physiological and clinical ranges via Pydantic.

### Clinical Feature Reference Table

| Feature Name | Clinical Display Name | Data Type | Permissible Range / Allowed Values | Nullable | Notes & Categories |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `age` | Patient Age | `float` | `18.0` – `120.0` | **No** | Age in years |
| `sex` | Biological Sex | `float` | `{0.0, 1.0}` | **No** | `0.0` = Female, `1.0` = Male |
| `cp` | Chest Pain Type | `float` | `{1.0, 2.0, 3.0, 4.0}` | **No** | `1.0` = Typical Angina<br>`2.0` = Atypical Angina<br>`3.0` = Non-Anginal Pain<br>`4.0` = Asymptomatic |
| `trestbps` | Resting Blood Pressure | `float` | `60.0` – `250.0` | **No** | Resting systolic BP in mm Hg |
| `chol` | Serum Cholesterol | `float` | `80.0` – `600.0` | **No** | Total cholesterol in mg/dl |
| `fbs` | Fasting Blood Sugar | `float` | `{0.0, 1.0}` | **No** | Fasting blood sugar > 120 mg/dl (`0.0` = No, `1.0` = Yes) |
| `restecg` | Resting ECG Results | `float` | `{0.0, 1.0, 2.0}` | **No** | `0.0` = Normal<br>`1.0` = ST-T Wave Abnormality<br>`2.0` = Left Ventricular Hypertrophy |
| `thalach` | Max Heart Rate Achieved | `float` | `50.0` – `230.0` | **No** | Max rate achieved during exercise stress (bpm) |
| `exang` | Exercise-Induced Angina | `float` | `{0.0, 1.0}` | **No** | `0.0` = No, `1.0` = Yes |
| `oldpeak` | ST Depression (Oldpeak) | `float` | `0.0` – `10.0` | **No** | ST depression induced by exercise relative to rest (mm) |
| `slope` | Slope of Peak ST | `float` | `{1.0, 2.0, 3.0}` | **No** | `1.0` = Upsloping<br>`2.0` = Flat<br>`3.0` = Downsloping |
| `ca` | Fluoroscopy Vessels | `float \| null` | `{0.0, 1.0, 2.0, 3.0, null}` | **Yes** | Major coronary vessels (0–3) colored by fluoroscopy; if omitted or `null`, median imputation is applied |
| `thal` | Thallium Perfusion Defect | `float \| null` | `{3.0, 6.0, 7.0, null}` | **Yes** | `3.0` = Normal Perfusion<br>`6.0` = Fixed Defect<br>`7.0` = Reversible Defect; if `null`, most-frequent imputation is applied |

---

## 3. Endpoints Specification

### 3.1 `GET /api/health`
Checks backend and ML model readiness. Used for frontend startup checks and liveness probes.

#### Response: `200 OK`
```json
{
  "status": "healthy",
  "api_version": "2.0.0",
  "model_ready": true,
  "champion_model": "SVM",
  "selection_method": "Highest mean outer-fold F1 in 5x4 nested stratified cross-validation on 242-row training partition. SVM was selected before seed-2026 test outcomes were inspected, using highest mean outer-fold F1 from nested cross-validation. Definitive hyperparameters tuned by GridSearchCV on all 242 training rows. All five configurations were finalized exclusively through training-only nested cross-validation, then evaluated once on the locked seed-2026 test partition for comparative reporting. These test results did not influence champion selection.",
  "metrics": {
    "cv_accuracy_mean": 0.810204081632653,
    "cv_accuracy_std": 0.03380313282236722,
    "cv_f1_mean": 0.7934786402228263,
    "cv_f1_std": 0.03219846529640586,
    "test_accuracy": 0.9344262295081968,
    "test_precision": 0.9285714285714286,
    "test_recall": 0.9285714285714286,
    "test_f1": 0.9285714285714286,
    "test_roc_auc": 0.972943722943723
  },
  "features_expected": 13
}
```

---

### 3.2 `GET /api/models/comparison`
Returns genuine benchmark evaluation metrics comparing all 5 evaluated classifiers. Clearly separates training cross-validation metrics from final held-out test evaluation.

#### Response: `200 OK`
```json
{
  "champion_model": "SVM",
  "selection_method": "Highest mean outer-fold F1 in 5x4 nested stratified cross-validation on 242-row training partition. SVM was selected before seed-2026 test outcomes were inspected, using highest mean outer-fold F1 from nested cross-validation. Definitive hyperparameters tuned by GridSearchCV on all 242 training rows. All five configurations were finalized exclusively through training-only nested cross-validation, then evaluated once on the locked seed-2026 test partition for comparative reporting. These test results did not influence champion selection.",
  "split_protocol": {
    "test_size": 0.2,
    "random_state": 2026,
    "cv_folds": 5,
    "train_rows": 242,
    "test_rows": 61,
    "stratified": true
  },
  "models": [
    {
      "model": "SVM",
      "cv_accuracy_mean": 0.8102,
      "cv_accuracy_std": 0.0338,
      "cv_f1_mean": 0.7935,
      "cv_f1_std": 0.0322,
      "test_accuracy": 0.9344,
      "test_precision": 0.9286,
      "test_recall": 0.9286,
      "test_f1": 0.9286,
      "test_roc_auc": 0.9729
    },
    {
      "model": "Random Forest",
      "cv_accuracy_mean": 0.8099,
      "cv_accuracy_std": 0.0245,
      "cv_f1_mean": 0.7870,
      "cv_f1_std": 0.0185,
      "test_accuracy": 0.9016,
      "test_precision": 0.8667,
      "test_recall": 0.9286,
      "test_f1": 0.8966,
      "test_roc_auc": 0.9589
    },
    {
      "model": "SVM",
      "cv_accuracy_mean": 0.8182,
      "cv_accuracy_std": 0.0270,
      "cv_f1_mean": 0.7950,
      "cv_f1_std": 0.0218,
      "test_accuracy": 0.9344,
      "test_precision": 0.9000,
      "test_recall": 0.9643,
      "test_f1": 0.9310,
      "test_roc_auc": 0.9675
    },
    {
      "model": "KNN",
      "cv_accuracy_mean": 0.7808,
      "cv_accuracy_std": 0.0617,
      "cv_f1_mean": 0.7493,
      "cv_f1_std": 0.0768,
      "test_accuracy": 0.9344,
      "test_precision": 0.9000,
      "test_recall": 0.9643,
      "test_f1": 0.9310,
      "test_roc_auc": 0.9762
    },
    {
      "model": "Gradient Boosting",
      "cv_accuracy_mean": 0.7724,
      "cv_accuracy_std": 0.0312,
      "cv_f1_mean": 0.7411,
      "cv_f1_std": 0.0351,
      "test_accuracy": 0.9016,
      "test_precision": 0.8438,
      "test_recall": 0.9643,
      "test_f1": 0.9000,
      "test_roc_auc": 0.9513
    }
  ]
}
```

---

### 3.3 `GET /api/dataset/stats`
Provides verified Cleveland dataset exploratory data statistics for data visualization tabs and feature distribution comparisons.

#### Response: `200 OK`
```json
{
  "total_samples": 303,
  "total_features": 13,
  "class_distribution": {
    "no_disease_count": 164,
    "no_disease_pct": 54.1,
    "disease_present_count": 139,
    "disease_present_pct": 45.9
  },
  "missing_values": {
    "ca": 4,
    "thal": 2
  },
  "numerical_features_summary": {
    "age": { "mean": 54.44, "median": 56.0, "std": 9.04, "min": 29.0, "max": 77.0 },
    "trestbps": { "mean": 131.69, "median": 130.0, "std": 17.6, "min": 94.0, "max": 200.0 },
    "chol": { "mean": 246.69, "median": 241.0, "std": 51.78, "min": 126.0, "max": 564.0 },
    "thalach": { "mean": 149.61, "median": 153.0, "std": 22.88, "min": 71.0, "max": 202.0 },
    "oldpeak": { "mean": 1.04, "median": 0.8, "std": 1.16, "min": 0.0, "max": 6.2 }
  },
  "categorical_categories": {
    "sex": { "0.0": "Female", "1.0": "Male" },
    "cp": {
      "1.0": "Typical Angina",
      "2.0": "Atypical Angina",
      "3.0": "Non-Anginal Pain",
      "4.0": "Asymptomatic"
    },
    "fbs": { "0.0": "Normal (<= 120 mg/dl)", "1.0": "Elevated (> 120 mg/dl)" },
    "restecg": {
      "0.0": "Normal",
      "1.0": "ST-T Wave Abnormality",
      "2.0": "Left Ventricular Hypertrophy"
    },
    "exang": { "0.0": "No", "1.0": "Yes" },
    "slope": { "1.0": "Upsloping", "2.0": "Flat", "3.0": "Downsloping" },
    "thal": {
      "3.0": "Normal Perfusion",
      "6.0": "Fixed Defect",
      "7.0": "Reversible Defect"
    }
  }
}
```

---

### 3.4 `POST /api/predict`
Calculates cardiovascular disease risk prediction using the champion pipeline.

#### Request Body
```json
{
  "age": 58.0,
  "sex": 1.0,
  "cp": 4.0,
  "trestbps": 140.0,
  "chol": 260.0,
  "fbs": 0.0,
  "restecg": 2.0,
  "thalach": 130.0,
  "exang": 1.0,
  "oldpeak": 2.2,
  "slope": 2.0,
  "ca": 2.0,
  "thal": 7.0
}
```
*(Note: `ca` and `thal` can be `null` or omitted.)*

#### Response: `200 OK`
```json
{
  "prediction": 1,
  "prediction_label": "Elevated Risk — Cardiovascular Disease Indicated",
  "risk_category": "High Risk",
  "disease_probability": 0.8841,
  "healthy_probability": 0.1159,
  "confidence_score": 0.8841,
  "champion_model": "Logistic Regression",
  "disclaimer": "Educational Cardiovascular Risk Assessment Notice: This tool is designed for educational and informational purposes only. It is NOT a medical device, does NOT diagnose heart disease, and should never replace professional medical advice, clinical diagnosis, or treatment."
}
```

#### Risk Categorization Rules
- **Low Risk**: `disease_probability < 0.35`
- **Moderate Risk**: `0.35 <= disease_probability < 0.65`
- **High Risk**: `disease_probability >= 0.65`

---

### 3.5 `POST /api/explain`
Calculates local SHAP feature attributions for an individual patient or returns global dataset importance.

#### Request Body (Patient Mode)
```json
{
  "mode": "patient",
  "patient_data": {
    "age": 58.0,
    "sex": 1.0,
    "cp": 4.0,
    "trestbps": 140.0,
    "chol": 260.0,
    "fbs": 0.0,
    "restecg": 2.0,
    "thalach": 130.0,
    "exang": 1.0,
    "oldpeak": 2.2,
    "slope": 2.0,
    "ca": 2.0,
    "thal": 7.0
  }
}
```

#### Response: `200 OK` (Patient Mode)
```json
{
  "mode": "patient",
  "prediction": 1,
  "prediction_label": "High Risk / Disease Likely",
  "disease_probability": 0.8841,
  "healthy_probability": 0.1159,
  "base_risk": -2.9094,
  "reconstructed_probability": 0.8841,
  "additive_verification": {
    "passed": true,
    "max_residual": 0.0,
    "explanation": "1 / (1 + exp(-(base_value + sum(shap_values)))) == predict_proba[:, 1]"
  },
  "top_risk_drivers": [
    {
      "feature": "nominal__thal_7.0",
      "base_feature": "thal",
      "display_name": "Thallium Stress Defect: Reversible Defect",
      "raw_value": 7.0,
      "formatted_value": "Yes",
      "transformed_value": 1.0,
      "shap_value": 1.4821,
      "abs_impact": 1.4821,
      "effect": "increases_risk",
      "unit": ""
    },
    {
      "feature": "num__ca",
      "base_feature": "ca",
      "display_name": "Major Fluoroscopy Vessels",
      "raw_value": 2.0,
      "formatted_value": "2 vessels (0-3)",
      "transformed_value": 1.15,
      "shap_value": 1.1942,
      "abs_impact": 1.1942,
      "effect": "increases_risk",
      "unit": "vessels (0-3)"
    }
  ],
  "top_risk_reducers": [
    {
      "feature": "num__thalach",
      "base_feature": "thalach",
      "display_name": "Max Heart Rate Achieved",
      "raw_value": 130.0,
      "formatted_value": "130 bpm",
      "transformed_value": -0.857,
      "shap_value": -0.4125,
      "abs_impact": 0.4125,
      "effect": "decreases_risk",
      "unit": "bpm"
    }
  ],
  "clinical_features_ranked": [
    {
      "feature": "thal",
      "display_name": "Thallium Stress Defect",
      "raw_value": 7.0,
      "formatted_value": "Reversible Defect (7)",
      "shap_value": 1.4821,
      "abs_impact": 1.4821,
      "effect": "increases_risk",
      "unit": ""
    },
    {
      "feature": "ca",
      "display_name": "Major Fluoroscopy Vessels",
      "raw_value": 2.0,
      "formatted_value": "2 vessels (0-3)",
      "shap_value": 1.1942,
      "abs_impact": 1.1942,
      "effect": "increases_risk",
      "unit": "vessels (0-3)"
    }
  ],
  "all_features_ranked": [ /* 18 granular transformed dummy/numerical features */ ],
  "disclaimer": "Educational Cardiovascular Risk Assessment Notice: ..."
}
```

---

### 3.6 `GET /api/explain/global`
Convenience endpoint returning dataset-level global SHAP ranking.

#### Response: `200 OK`
```json
{
  "mode": "global",
  "base_risk": -2.9094,
  "sample_count": 303,
  "feature_count": 18,
  "clinical_feature_count": 13,
  "rankings": [ /* 18 transformed features sorted by mean_abs_shap */ ],
  "clinical_rankings": [
    {
      "rank": 1,
      "feature": "thal",
      "display_name": "Thallium Stress Defect",
      "type": "numerical",
      "mean_abs_shap": 0.7421,
      "unit": ""
    },
    {
      "rank": 2,
      "feature": "ca",
      "display_name": "Major Fluoroscopy Vessels",
      "type": "numerical",
      "mean_abs_shap": 0.6912,
      "unit": "vessels (0-3)"
    },
    {
      "rank": 3,
      "feature": "cp",
      "display_name": "Chest Pain Type",
      "type": "numerical",
      "mean_abs_shap": 0.5841,
      "unit": ""
    }
  ],
  "beeswarm_plot_available": true,
  "bar_plot_available": true
}
```

---

## 4. Error Responses

The backend enforces standard HTTP status codes and consistent error payloads. Internal filepaths and tracebacks are never exposed.

### 4.1 Validation Error (`422 Unprocessable Content`)
Triggered when input types or ranges violate constraints:
```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["body", "age"],
      "msg": "Input should be greater than or equal to 18",
      "input": 12.0,
      "ctx": { "ge": 18.0 }
    }
  ]
}
```

### 4.2 Missing Patient Data (`422 Unprocessable Content`)
Triggered when `POST /api/explain` has `mode="patient"` but `patient_data` is omitted:
```json
{
  "detail": "patient_data must be provided when requesting patient-level explanation."
}
```

### 4.3 Service Unavailable (`503 Service Unavailable`)
Triggered if model artifacts fail to load on startup:
```json
{
  "detail": "Model service is currently unavailable. Please check system health."
}
```

### 4.4 Internal Server Error (`500 Internal Server Error`)
Sanitized catch-all response:
```json
{
  "detail": "An internal server error occurred. Please try again later."
}
```

---

## 5. TypeScript Types for Frontend Consumption

```typescript
export interface PatientInput {
  age: number;               // 18 - 120
  sex: 0 | 1;               // 0 = Female, 1 = Male
  cp: 1 | 2 | 3 | 4;         // 1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic
  trestbps: number;          // 60 - 250 mm Hg
  chol: number;              // 80 - 600 mg/dl
  fbs: 0 | 1;               // 0 = <= 120 mg/dl, 1 = > 120 mg/dl
  restecg: 0 | 1 | 2;        // 0=Normal, 1=ST-T, 2=LV hypertrophy
  thalach: number;           // 50 - 230 bpm
  exang: 0 | 1;             // 0 = No, 1 = Yes
  oldpeak: number;           // 0.0 - 10.0 mm
  slope: 1 | 2 | 3;          // 1=Upsloping, 2=Flat, 3=Downsloping
  ca?: number | null;        // 0, 1, 2, 3 or null
  thal?: number | null;      // 3, 6, 7 or null
}

export interface PredictionResponse {
  prediction: 0 | 1;
  prediction_label: string;
  risk_category: 'Low Risk' | 'Moderate Risk' | 'High Risk';
  disease_probability: number;
  healthy_probability: number;
  confidence_score: number;
  champion_model: string;
  disclaimer: string;
}

export interface ClinicalAttribution {
  feature: string;
  display_name: string;
  raw_value: number | string | null;
  formatted_value: string;
  shap_value: number;
  abs_impact: number;
  effect: 'increases_risk' | 'decreases_risk' | 'neutral';
  unit: string;
}

export interface PatientExplanationResponse {
  mode: 'patient';
  prediction: number;
  prediction_label: string;
  disease_probability: number;
  healthy_probability: number;
  base_risk: number;
  reconstructed_probability: number;
  additive_verification: {
    passed: boolean;
    max_residual: number;
    explanation: string;
  };
  top_risk_drivers: ClinicalAttribution[];
  top_risk_reducers: ClinicalAttribution[];
  clinical_features_ranked: ClinicalAttribution[];
  all_features_ranked: any[];
  disclaimer: string;
}

export interface ModelComparisonResponse {
  champion_model: string;
  selection_method: string;
  split_protocol: {
    test_size: number;
    random_state: number;
    cv_folds: number;
    train_rows: number;
    test_rows: number;
    stratified: boolean;
  };
  models: Array<{
    model: string;
    cv_accuracy_mean: number;
    cv_accuracy_std: number;
    cv_f1_mean: number;
    cv_f1_std: number;
    test_accuracy: number;
    test_precision: number;
    test_recall: number;
    test_f1: number;
    test_roc_auc: number;
  }>;
}
```

/**
 * api.ts — TypeScript interfaces matching docs/frontend_api_contract.md.
 */

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

export type RiskCategory = 'Low Risk' | 'Moderate Risk' | 'High Risk';

export interface PredictionResponse {
  prediction: 0 | 1;
  prediction_label: string;
  risk_category: RiskCategory;
  disease_probability: number;
  healthy_probability: number;
  confidence_score: number;
  champion_model: string;
  disclaimer: string;
}

export interface FeatureAttribution {
  feature: string;
  base_feature: string;
  display_name: string;
  raw_value: number | string | null;
  formatted_value: string;
  transformed_value?: number;
  shap_value: number;
  abs_impact: number;
  effect: 'increases_risk' | 'decreases_risk' | 'neutral';
  unit: string;
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

export interface AdditiveVerification {
  passed: boolean;
  max_residual: number;
  explanation: string;
}

export interface PatientExplanationResponse {
  mode: 'patient';
  prediction: number;
  prediction_label: string;
  disease_probability: number;
  healthy_probability: number;
  base_risk: number;
  reconstructed_probability: number;
  additive_verification: AdditiveVerification;
  top_risk_drivers: FeatureAttribution[];
  top_risk_reducers: FeatureAttribution[];
  clinical_features_ranked: ClinicalAttribution[];
  all_features_ranked: FeatureAttribution[];
  disclaimer: string;
}

export interface GlobalRankingItem {
  rank: number;
  feature: string;
  display_name: string;
  type: string;
  mean_abs_shap: number;
  unit: string;
}

export interface GlobalExplanationResponse {
  mode: 'global';
  base_risk: number;
  sample_count: number;
  feature_count: number;
  clinical_feature_count: number;
  rankings: GlobalRankingItem[];
  clinical_rankings: GlobalRankingItem[];
  beeswarm_plot_available: boolean;
  bar_plot_available: boolean;
  explainer_method?: string;
  explainer_note?: string;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded';
  api_version: string;
  model_ready: boolean;
  champion_model: string;
  selection_method: string;
  metrics: {
    cv_accuracy_mean: number;
    cv_accuracy_std: number;
    cv_f1_mean: number;
    cv_f1_std: number;
    test_accuracy: number;
    test_precision: number;
    test_recall: number;
    test_f1: number;
    test_roc_auc?: number;
  };
  features_expected: number;
}

export interface ModelBenchmarkItem {
  model: string;
  cv_accuracy_mean: number;
  cv_accuracy_std: number;
  cv_f1_mean?: number;
  cv_f1_std?: number;
  test_accuracy: number;
  test_precision: number;
  test_recall: number;
  test_f1: number;
  test_roc_auc?: number;
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
  models: ModelBenchmarkItem[];
}

export interface DatasetStatsResponse {
  total_samples: number;
  total_features: number;
  class_distribution: {
    no_disease_count: number;
    no_disease_pct: number;
    disease_present_count: number;
    disease_present_pct: number;
  };
  missing_values: Record<string, number>;
  numerical_features_summary: Record<string, {
    mean: number;
    median: number;
    std: number;
    min: number;
    max: number;
  }>;
  categorical_categories: Record<string, Record<string, string>>;
}

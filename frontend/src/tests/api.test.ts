import { describe, it, expect, vi, beforeEach } from 'vitest';
import { apiService, ApiError } from '../services/api';
import type { PatientInput } from '../types/api';

describe('Frontend API Service Client', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  const samplePatient: PatientInput = {
    age: 58.0,
    sex: 1.0,
    cp: 4.0,
    trestbps: 140.0,
    chol: 260.0,
    fbs: 0.0,
    restecg: 2.0,
    thalach: 130.0,
    exang: 1.0,
    oldpeak: 2.2,
    slope: 2.0,
    ca: 2.0,
    thal: 7.0,
  };

  it('correctly maps successful health response', async () => {
    const mockHealth = {
      status: 'healthy',
      api_version: '2.0.0',
      model_ready: true,
      champion_model: 'SVM',
      selection_method: 'highest mean outer-fold CV F1',
      metrics: { cv_accuracy_mean: 0.8102, test_accuracy: 0.9344 },
      features_expected: 13,
    };

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => mockHealth,
    } as Response);

    const health = await apiService.getHealth();
    expect(health.status).toBe('healthy');
    expect(health.champion_model).toBe('SVM');
    expect(health.features_expected).toBe(13);
  });

  it('throws ApiError with detail message on 422 error', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: false,
      status: 422,
      json: async () => ({
        detail: [{ loc: ['body', 'age'], msg: 'Input should be greater than or equal to 18' }],
      }),
    } as Response);

    await expect(apiService.predictRisk(samplePatient)).rejects.toThrow(ApiError);
  });

  it('correctly serializes patient prediction input and receives risk output', async () => {
    const mockPrediction = {
      prediction: 1,
      prediction_label: 'Elevated Risk — Cardiovascular Disease Indicated',
      risk_category: 'High Risk',
      disease_probability: 0.8841,
      healthy_probability: 0.1159,
      confidence_score: 0.8841,
      champion_model: 'SVM',
      disclaimer: 'Educational notice only.',
    };

    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => mockPrediction,
    } as Response);

    const pred = await apiService.predictRisk(samplePatient);
    expect(pred.prediction).toBe(1);
    expect(pred.risk_category).toBe('High Risk');
    expect(pred.disease_probability).toBe(0.8841);

    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringContaining('/api/predict'),
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(samplePatient),
      })
    );
  });

  it('correctly requests local patient SHAP explanation', async () => {
    const mockExplanation = {
      mode: 'patient',
      prediction: 1,
      prediction_label: 'High Risk',
      disease_probability: 0.8841,
      healthy_probability: 0.1159,
      base_risk: -2.9094,
      reconstructed_probability: 0.8841,
      additive_verification: { passed: true, max_residual: 0.0, explanation: 'verified' },
      top_risk_drivers: [],
      top_risk_reducers: [],
      clinical_features_ranked: [],
      all_features_ranked: [],
      disclaimer: 'Educational notice only.',
    };

    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => mockExplanation,
    } as Response);

    const exp = await apiService.explainPatient(samplePatient);
    expect(exp.mode).toBe('patient');
    expect(exp.additive_verification.passed).toBe(true);

    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringContaining('/api/explain'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ mode: 'patient', patient_data: samplePatient }),
      })
    );
  });

  it('correctly fetches model comparison benchmark data', async () => {
    const mockComparison = {
      champion_model: 'SVM',
      selection_method: 'highest mean outer-fold CV F1',
      split_protocol: { test_size: 0.2, random_state: 2026, stratify: true },
      models: [
        {
          model: 'SVM',
          cv_accuracy_mean: 0.8102,
          cv_accuracy_std: 0.0338,
          cv_f1_mean: 0.7935,
          test_accuracy: 0.9344,
          test_precision: 0.9286,
          test_recall: 0.9286,
          test_f1: 0.9286,
          test_roc_auc: 0.9729,
        },
      ],
    };

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => mockComparison,
    } as Response);

    const data = await apiService.getModelComparison();
    expect(data.champion_model).toBe('SVM');
    expect(data.models).toHaveLength(1);
    expect(data.models[0].cv_f1_mean).toBe(0.7935);
  });

  it('correctly fetches dataset stats', async () => {
    const mockStats = {
      total_samples: 303,
      total_features: 13,
      class_distribution: {
        no_disease_count: 164,
        disease_present_count: 139,
        no_disease_pct: 54.1,
        disease_present_pct: 45.9,
      },
      missing_values: { ca: 4, thal: 2 },
    };

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => mockStats,
    } as Response);

    const stats = await apiService.getDatasetStats();
    expect(stats.total_samples).toBe(303);
    expect(stats.class_distribution.no_disease_count).toBe(164);
  });
});


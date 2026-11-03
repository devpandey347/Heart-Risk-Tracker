/**
 * api.ts — HTTP client for Heart Risk Tracker backend API.
 */

import type {
  DatasetStatsResponse,
  GlobalExplanationResponse,
  HealthResponse,
  ModelComparisonResponse,
  PatientExplanationResponse,
  PatientInput,
  PredictionResponse,
} from '../types/api';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

export class ApiError extends Error {
  status: number;
  data: any;

  constructor(message: string, status: number, data?: any) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorDetail = 'API request failed';
    try {
      const errJson = await res.json();
      if (typeof errJson.detail === 'string') {
        errorDetail = errJson.detail;
      } else if (Array.isArray(errJson.detail)) {
        errorDetail = errJson.detail.map((d: any) => `${d.loc?.join('.') || 'field'}: ${d.msg}`).join('; ');
      }
    } catch {
      errorDetail = `HTTP ${res.status}: ${res.statusText}`;
    }
    throw new ApiError(errorDetail, res.status);
  }
  return res.json() as Promise<T>;
}

export const apiService = {
  /**
   * System health and model operational status
   */
  async getHealth(): Promise<HealthResponse> {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    return handleResponse<HealthResponse>(res);
  },

  /**
   * Genuine benchmark comparison of all 5 classifiers
   */
  async getModelComparison(): Promise<ModelComparisonResponse> {
    const res = await fetch(`${API_BASE_URL}/api/models/comparison`);
    return handleResponse<ModelComparisonResponse>(res);
  },

  /**
   * Verified Cleveland dataset distributions & statistics
   */
  async getDatasetStats(): Promise<DatasetStatsResponse> {
    const res = await fetch(`${API_BASE_URL}/api/dataset/stats`);
    return handleResponse<DatasetStatsResponse>(res);
  },

  /**
   * Patient cardiovascular risk inference
   */
  async predictRisk(patientData: PatientInput): Promise<PredictionResponse> {
    const res = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(patientData),
    });
    return handleResponse<PredictionResponse>(res);
  },

  /**
   * Local SHAP patient prediction explanation
   */
  async explainPatient(patientData: PatientInput): Promise<PatientExplanationResponse> {
    const res = await fetch(`${API_BASE_URL}/api/explain`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        mode: 'patient',
        patient_data: patientData,
      }),
    });
    return handleResponse<PatientExplanationResponse>(res);
  },

  /**
   * Global SHAP feature importance rankings
   */
  async getGlobalExplanation(): Promise<GlobalExplanationResponse> {
    const res = await fetch(`${API_BASE_URL}/api/explain/global`);
    return handleResponse<GlobalExplanationResponse>(res);
  },
};

import React, { useState } from 'react';
import type { PatientExplanationResponse, PatientInput, PredictionResponse } from '../types/api';
import { apiService, ApiError } from '../services/api';
import { PatientForm } from '../components/estimation/PatientForm';
import { PredictionResult } from '../components/estimation/PredictionResult';
import { PatientExplanation } from '../components/explainability/PatientExplanation';
import { AlertCircle, Stethoscope } from 'lucide-react';

interface EstimationViewProps {
  onExplanationGenerated?: (exp: PatientExplanationResponse) => void;
}

export const EstimationView: React.FC<EstimationViewProps> = ({
  onExplanationGenerated,
}) => {
  const [loadingPredict, setLoadingPredict] = useState<boolean>(false);
  const [loadingExplain, setLoadingExplain] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [patientData, setPatientData] = useState<PatientInput | null>(null);
  const [patientExplanation, setPatientExplanation] = useState<PatientExplanationResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleFormSubmit = async (data: PatientInput) => {
    setLoadingPredict(true);
    setErrorMessage(null);
    setPatientData(data);
    setPatientExplanation(null);

    try {
      const pred = await apiService.predictRisk(data);
      setPrediction(pred);
    } catch (err: any) {
      if (err instanceof ApiError) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage('Failed to connect to the prediction API. Ensure the backend is running.');
      }
    } finally {
      setLoadingPredict(false);
    }
  };

  const handleExplainRequested = async () => {
    if (!patientData) return;

    setLoadingExplain(true);
    setErrorMessage(null);

    try {
      const exp = await apiService.explainPatient(patientData);
      setPatientExplanation(exp);
      if (onExplanationGenerated) {
        onExplanationGenerated(exp);
      }
    } catch (err: any) {
      if (err instanceof ApiError) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage('Failed to generate SHAP explanation. Please try again.');
      }
    } finally {
      setLoadingExplain(false);
    }
  };

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      {/* View Header */}
      <div>
        <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900 flex items-center gap-2.5">
          <Stethoscope className="w-6 h-6 text-teal-600" />
          <span>Patient Risk Estimation</span>
        </h2>
        <p className="text-xs sm:text-sm text-slate-500 mt-1">
          Enter 13 clinical biomarkers to compute estimated cardiovascular disease risk probability using the verified champion pipeline.
        </p>
      </div>

      {/* Error Callout */}
      {errorMessage && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-800 text-sm flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-rose-600 flex-shrink-0 mt-0.5" />
          <div>
            <h5 className="font-semibold text-rose-900">Inference Request Error</h5>
            <p className="text-xs text-rose-700 mt-0.5">{errorMessage}</p>
          </div>
        </div>
      )}

      {/* Input Form */}
      <PatientForm
        onSubmit={handleFormSubmit}
        loading={loadingPredict}
      />

      {/* Result Section */}
      {prediction && (
        <div className="space-y-6 pt-2">
          <PredictionResult
            result={prediction}
            onExplainClick={handleExplainRequested}
            loadingExplain={loadingExplain}
          />

          {/* Inline Patient SHAP Explanation if generated */}
          {patientExplanation && (
            <div className="pt-2 animate-fadeIn">
              <PatientExplanation
                explanation={patientExplanation}
                loading={loadingExplain}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

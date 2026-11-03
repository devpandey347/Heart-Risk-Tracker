import React, { useState } from 'react';
import type { GlobalExplanationResponse, PatientExplanationResponse } from '../types/api';
import { PatientExplanation } from '../components/explainability/PatientExplanation';
import { GlobalImportance } from '../components/explainability/GlobalImportance';
import { Sparkles, Globe, UserCheck } from 'lucide-react';

interface ExplainabilityViewProps {
  patientExplanation: PatientExplanationResponse | null;
  globalData: GlobalExplanationResponse | null;
  loadingGlobal: boolean;
  globalError?: string | null;
  onNavigateToEstimation: () => void;
}

export const ExplainabilityView: React.FC<ExplainabilityViewProps> = ({
  patientExplanation,
  globalData,
  loadingGlobal,
  globalError,
  onNavigateToEstimation,
}) => {
  const [activeTab, setActiveTab] = useState<'patient' | 'global'>('global');

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* View Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900 flex items-center gap-2.5">
            <Sparkles className="w-6 h-6 text-teal-600" />
            <span>SHAP Explainability & Model Interpretability</span>
          </h2>
          <p className="text-xs sm:text-sm text-slate-500 mt-1">
            Understand how clinical biomarkers drive individual patient predictions and rank across the broader population.
          </p>
        </div>

        {/* View Switcher Tabs */}
        <div className="flex items-center p-1 bg-slate-100 rounded-xl border border-slate-200 self-start sm:self-auto">
          <button
            onClick={() => setActiveTab('global')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'global'
                ? 'bg-white text-teal-900 shadow-2xs border border-slate-200/60'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Globe className="w-3.5 h-3.5 text-teal-600" />
            <span>Global Importance</span>
          </button>

          <button
            onClick={() => setActiveTab('patient')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'patient'
                ? 'bg-white text-teal-900 shadow-2xs border border-slate-200/60'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <UserCheck className="w-3.5 h-3.5 text-teal-600" />
            <span>Patient Explanation</span>
            {patientExplanation && (
              <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
            )}
          </button>
        </div>
      </div>

      {/* Global Tab */}
      {activeTab === 'global' && (
        <GlobalImportance
          globalData={globalData}
          loading={loadingGlobal}
          error={globalError}
        />
      )}

      {/* Patient Tab */}
      {activeTab === 'patient' && (
        <div>
          {patientExplanation ? (
            <PatientExplanation
              explanation={patientExplanation}
              loading={false}
            />
          ) : (
            <div className="bg-white rounded-2xl border border-slate-200/80 p-10 text-center space-y-4 shadow-xs">
              <div className="w-12 h-12 rounded-2xl bg-teal-50 text-teal-700 flex items-center justify-center mx-auto">
                <Sparkles className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <h4 className="text-base font-semibold text-slate-800">No Patient Profile Evaluated Yet</h4>
                <p className="text-xs text-slate-500 max-w-md mx-auto leading-relaxed">
                  Local SHAP attributions require patient input features. Complete the Risk Estimation form first to inspect how each biomarker impacted the assessment.
                </p>
              </div>
              <button
                onClick={onNavigateToEstimation}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-teal-700 hover:bg-teal-800 text-white font-medium text-xs shadow-xs transition-all"
              >
                Go to Risk Estimation Form
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

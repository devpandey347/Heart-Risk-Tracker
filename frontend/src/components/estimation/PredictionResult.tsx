import React from 'react';
import type { PredictionResponse } from '../../types/api';
import { RiskBadge } from '../common/RiskBadge';
import { DisclaimerBanner } from '../common/DisclaimerBanner';
import { Sparkles, Activity, Cpu } from 'lucide-react';

interface PredictionResultProps {
  result: PredictionResponse | null;
  onExplainClick: () => void;
  loadingExplain?: boolean;
}

export const PredictionResult: React.FC<PredictionResultProps> = ({
  result,
  onExplainClick,
  loadingExplain = false,
}) => {
  if (!result) return null;

  const probPercent = (result.disease_probability * 100).toFixed(1);
  const healthyPercent = (result.healthy_probability * 100).toFixed(1);

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-sm space-y-6">
      {/* Result Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div>
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">Inference Output</span>
          <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
            <Activity className="w-5 h-5 text-teal-600" />
            Cardiovascular Risk Assessment Result
          </h3>
        </div>

        <RiskBadge category={result.risk_category} size="lg" />
      </div>

      {/* Probability Gauge & Primary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
        {/* Progress Bar & Probability */}
        <div className="md:col-span-2 space-y-4">
          <div className="flex items-baseline justify-between">
            <div>
              <span className="text-xs text-slate-500 font-medium">Estimated Disease Probability</span>
              <div className="text-3xl sm:text-4xl font-extrabold font-mono text-slate-900 tracking-tight mt-0.5">
                {probPercent}%
              </div>
            </div>
            <div className="text-right">
              <span className="text-xs text-slate-500 font-medium">Healthy Probability</span>
              <div className="text-xl sm:text-2xl font-bold font-mono text-emerald-700 tracking-tight mt-0.5">
                {healthyPercent}%
              </div>
            </div>
          </div>

          {/* Visual Risk Probability Bar */}
          <div className="space-y-1.5">
            <div className="w-full h-3.5 bg-slate-100 rounded-full overflow-hidden p-0.5 border border-slate-200">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  result.risk_category === 'Low Risk'
                    ? 'bg-emerald-500'
                    : result.risk_category === 'Moderate Risk'
                    ? 'bg-amber-500'
                    : 'bg-rose-500'
                }`}
                style={{ width: `${Math.max(result.disease_probability * 100, 3)}%` }}
              />
            </div>
            <div className="flex justify-between text-[11px] text-slate-400 px-1 font-mono">
              <span>0% Low (&lt;35%)</span>
              <span>Moderate (35–65%)</span>
              <span>High (&ge;65%) 100%</span>
            </div>
          </div>

          {/* Prediction Label Callout */}
          <div className="p-3 bg-slate-50 rounded-xl border border-slate-200/80 text-xs text-slate-700 font-medium">
            <strong>Clinical Indicator:</strong> {result.prediction_label}
          </div>
        </div>

        {/* Model & Confidence Box */}
        <div className="bg-slate-50/80 border border-slate-200/80 rounded-xl p-4 space-y-3">
          <div>
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">Evaluating Pipeline</span>
            <div className="flex items-center gap-1.5 text-sm font-semibold text-slate-800 mt-0.5">
              <Cpu className="w-4 h-4 text-teal-600" />
              <span>{result.champion_model}</span>
            </div>
          </div>

          <div>
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">Model Confidence</span>
            <span className="text-lg font-bold font-mono text-slate-900">
              {(result.confidence_score * 100).toFixed(1)}%
            </span>
            <span className="text-[11px] text-slate-500 block">Maximum class posterior</span>
          </div>

          <button
            onClick={onExplainClick}
            disabled={loadingExplain}
            className="w-full inline-flex items-center justify-center gap-2 px-4 py-2 rounded-xl bg-teal-600 hover:bg-teal-700 text-white font-semibold text-xs transition-all shadow-xs focus:outline-none focus:ring-2 focus:ring-teal-500/30 disabled:opacity-50"
          >
            <Sparkles className="w-3.5 h-3.5 text-teal-200" />
            <span>{loadingExplain ? 'Analyzing SHAP...' : 'Explain This Prediction'}</span>
          </button>
        </div>
      </div>

      {/* Medical Disclaimer */}
      <DisclaimerBanner className="mt-4" />
    </div>
  );
};

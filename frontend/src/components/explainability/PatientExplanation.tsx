import React from 'react';
import type { PatientExplanationResponse } from '../../types/api';
import { Sparkles, TrendingUp, TrendingDown, CheckCircle2, AlertCircle, Info, BarChart2 } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';

interface PatientExplanationProps {
  explanation: PatientExplanationResponse | null;
  loading: boolean;
  error?: string | null;
}

export const PatientExplanation: React.FC<PatientExplanationProps> = ({
  explanation,
  loading,
  error,
}) => {
  if (loading) {
    return (
      <div className="bg-white rounded-2xl border border-slate-200/80 p-6 animate-pulse space-y-4">
        <div className="h-5 w-48 bg-slate-200 rounded"></div>
        <div className="h-32 bg-slate-100 rounded-xl"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-2xl border border-rose-200 p-6 text-rose-800 text-sm">
        Failed to generate SHAP explanation: {error}
      </div>
    );
  }

  if (!explanation) {
    return (
      <div className="bg-white rounded-2xl border border-slate-200/80 p-8 text-center text-slate-500 space-y-2">
        <Sparkles className="w-8 h-8 text-slate-300 mx-auto" />
        <h4 className="text-sm font-semibold text-slate-700">No Patient Explanation Active</h4>
        <p className="text-xs text-slate-400 max-w-sm mx-auto">
          Submit a patient profile in the Risk Estimation view and click "Explain This Prediction" to inspect patient-specific SHAP biomarker attributions.
        </p>
      </div>
    );
  }

  // Format waterfall / attribution chart data from clinical features
  const chartData = explanation.clinical_features_ranked.map((f) => ({
    name: f.display_name,
    shap: Number(f.shap_value.toFixed(4)),
    abs_impact: Number(f.abs_impact.toFixed(4)),
    formatted_value: f.formatted_value,
    effect: f.effect,
  }));

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div>
          <span className="text-[11px] font-semibold text-teal-700 uppercase tracking-wider block">Individual Patient Interpretability</span>
          <h3 className="text-base font-semibold text-slate-900 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-teal-600" />
            SHAP Local Feature Attributions
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Marginal contribution of each clinical measurement to the champion model's estimated risk score.
          </p>
        </div>

        {/* Additive Verification Pill */}
        <div className="flex items-center gap-2 self-start sm:self-auto">
          {explanation.additive_verification.passed ? (
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-50 text-emerald-800 border border-emerald-200">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              <span>SHAP Additivity Verified (Residual &le; 0.05)</span>
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-amber-50 text-amber-800 border border-amber-200">
              <AlertCircle className="w-3.5 h-3.5 text-amber-600" />
              <span>Residual: {explanation.additive_verification.max_residual.toExponential(2)}</span>
            </span>
          )}
        </div>
      </div>

      {/* Causation Warning Callout */}
      <div className="p-3.5 bg-slate-50 border border-slate-200/80 rounded-xl text-xs text-slate-600 flex items-start gap-2.5">
        <Info className="w-4 h-4 text-teal-600 flex-shrink-0 mt-0.5" />
        <p className="leading-relaxed">
          <strong>Important Interpretability Note:</strong> SHAP feature attributions describe how the mathematical model weighs inputs to form its statistical output. They represent algorithmic associations and <strong>do not establish medical causation or diagnosis</strong>.
        </p>
      </div>

      {/* Top Drivers & Reducers */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Top Risk Drivers */}
        <div className="bg-rose-50/40 border border-rose-200/60 rounded-xl p-4 space-y-3">
          <div className="flex items-center gap-2 text-rose-900 font-semibold text-xs uppercase tracking-wider">
            <TrendingUp className="w-4 h-4 text-rose-600" />
            <span>Top Risk Drivers (Elevate Risk)</span>
          </div>

          <div className="space-y-2">
            {explanation.top_risk_drivers.length === 0 ? (
              <p className="text-xs text-slate-400">No positive risk drivers identified.</p>
            ) : (
              explanation.top_risk_drivers.slice(0, 3).map((driver) => (
                <div key={driver.feature} className="p-2.5 bg-white border border-rose-200/80 rounded-lg flex items-center justify-between text-xs shadow-2xs">
                  <div>
                    <span className="font-semibold text-slate-800 block">{driver.display_name}</span>
                    <span className="text-[11px] text-slate-500">{driver.formatted_value}</span>
                  </div>
                  <span className="font-mono font-bold text-rose-700 bg-rose-50 px-2 py-0.5 rounded border border-rose-200">
                    +{driver.shap_value.toFixed(4)}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Top Risk Reducers */}
        <div className="bg-emerald-50/40 border border-emerald-200/60 rounded-xl p-4 space-y-3">
          <div className="flex items-center gap-2 text-emerald-900 font-semibold text-xs uppercase tracking-wider">
            <TrendingDown className="w-4 h-4 text-emerald-600" />
            <span>Top Risk Reducers (Mitigate Risk)</span>
          </div>

          <div className="space-y-2">
            {explanation.top_risk_reducers.length === 0 ? (
              <p className="text-xs text-slate-400">No mitigating risk factors identified.</p>
            ) : (
              explanation.top_risk_reducers.slice(0, 3).map((reducer) => (
                <div key={reducer.feature} className="p-2.5 bg-white border border-emerald-200/80 rounded-lg flex items-center justify-between text-xs shadow-2xs">
                  <div>
                    <span className="font-semibold text-slate-800 block">{reducer.display_name}</span>
                    <span className="text-[11px] text-slate-500">{reducer.formatted_value}</span>
                  </div>
                  <span className="font-mono font-bold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                    {reducer.shap_value.toFixed(4)}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Feature Attribution Bar Chart */}
      <div className="space-y-2 pt-2">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
          <BarChart2 className="w-3.5 h-3.5 text-teal-600" />
          Patient Attributions by Clinical Metric (SHAP Log-Odds Contribution)
        </h4>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 10, right: 30, left: 120, bottom: 10 }}
            >
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
              <XAxis type="number" tick={{ fontSize: 11, fill: '#64748b' }} />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fontSize: 11, fill: '#334155' }}
                interval={0}
                width={120}
              />
              <Tooltip
                formatter={(value: any, _: any, entry: any) => [
                  `${value > 0 ? '+' : ''}${value} impact (${entry.payload.formatted_value})`,
                  'SHAP Value',
                ]}
                contentStyle={{ borderRadius: '12px', border: '1px solid #e2e8f0', fontSize: '12px' }}
              />
              <ReferenceLine x={0} stroke="#94a3b8" />
              <Bar dataKey="shap" radius={[4, 4, 4, 4]}>
                {chartData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.shap > 0 ? '#ef4444' : '#10b981'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Full Clinical Table */}
      <div className="overflow-x-auto pt-2">
        <table className="w-full text-xs text-left text-slate-600 border border-slate-200/80 rounded-xl overflow-hidden">
          <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200/80">
            <tr>
              <th className="px-3 py-2.5">Clinical Feature</th>
              <th className="px-3 py-2.5">Patient Value</th>
              <th className="px-3 py-2.5">SHAP Attribution</th>
              <th className="px-3 py-2.5">Directional Impact</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {explanation.clinical_features_ranked.map((f) => (
              <tr key={f.feature} className="hover:bg-slate-50/60">
                <td className="px-3 py-2 text-slate-900 font-medium">{f.display_name}</td>
                <td className="px-3 py-2 font-mono text-slate-700">{f.formatted_value}</td>
                <td className="px-3 py-2 font-mono font-semibold">
                  <span className={f.shap_value > 0 ? 'text-rose-600' : 'text-emerald-600'}>
                    {f.shap_value > 0 ? `+${f.shap_value.toFixed(4)}` : f.shap_value.toFixed(4)}
                  </span>
                </td>
                <td className="px-3 py-2">
                  {f.effect === 'increases_risk' && (
                    <span className="text-[10px] font-semibold text-rose-700 bg-rose-50 border border-rose-200 px-2 py-0.5 rounded-full">
                      Increases Risk
                    </span>
                  )}
                  {f.effect === 'decreases_risk' && (
                    <span className="text-[10px] font-semibold text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full">
                      Decreases Risk
                    </span>
                  )}
                  {f.effect === 'neutral' && (
                    <span className="text-[10px] text-slate-400">Neutral</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

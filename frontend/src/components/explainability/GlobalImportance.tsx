import React from 'react';
import type { GlobalExplanationResponse } from '../../types/api';
import { Globe, Info } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface GlobalImportanceProps {
  globalData: GlobalExplanationResponse | null;
  loading: boolean;
  error?: string | null;
}

export const GlobalImportance: React.FC<GlobalImportanceProps> = ({
  globalData,
  loading,
  error,
}) => {
  if (loading) {
    return (
      <div className="bg-white rounded-2xl border border-slate-200/80 p-6 animate-pulse space-y-4">
        <div className="h-5 w-48 bg-slate-200 rounded"></div>
        <div className="h-64 bg-slate-100 rounded-xl"></div>
      </div>
    );
  }

  if (error || !globalData) {
    return (
      <div className="bg-white rounded-2xl border border-rose-200 p-6 text-rose-800 text-sm">
        Failed to load global SHAP importance: {error || 'Data unavailable'}
      </div>
    );
  }

  // Format chart data for Recharts (reverse for top-to-bottom layout in horizontal bars)
  const chartData = [...globalData.clinical_rankings].reverse().map((item) => ({
    name: item.display_name,
    mean_abs_shap: item.mean_abs_shap,
    rank: item.rank,
  }));

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div>
          <span className="text-[11px] font-semibold text-teal-700 uppercase tracking-wider block">Population-Level Interpretability</span>
          <h3 className="text-base font-semibold text-slate-900 flex items-center gap-2">
            <Globe className="w-4 h-4 text-teal-600" />
            Global Feature Importance (Mean |SHAP|)
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Average marginal impact of each clinical feature on the model's decision across all 303 cohort patients.
          </p>
        </div>

        <span className="text-xs font-mono bg-slate-100 text-slate-600 px-3 py-1 rounded-full font-medium self-start sm:self-auto">
          Cohort N=303 • Base Risk: {globalData.base_risk.toFixed(4)}
        </span>
      </div>

      {/* Info & Limitations Callout */}
      <div className="space-y-3">
        <div className="p-3.5 bg-slate-50 border border-slate-200/80 rounded-xl text-xs text-slate-600 flex items-start gap-2.5">
          <Info className="w-4 h-4 text-teal-600 flex-shrink-0 mt-0.5" />
          <div className="space-y-1 leading-relaxed">
            <p>
              <strong>Methodology & Clinical Scope:</strong> Features near the top have the largest average influence across the patient population. Thallium stress defects, major fluoroscopy vessels, and chest pain classification consistently emerge as the strongest primary contributors in the trained champion model ({globalData.explainer_method || 'SHAP'}).
            </p>
            {globalData.explainer_note && (
              <p className="text-slate-500 text-[11px]">
                <strong>Explainer Note:</strong> {globalData.explainer_note}
              </p>
            )}
            <p className="text-slate-500 text-[11px]">
              <strong>Limitations:</strong> Global feature importance reflects statistical associations learned from this specific training cohort (UCI Cleveland N=303). It does not represent population-wide epidemiological risk factors or establish medical causation.
            </p>
          </div>
        </div>
      </div>

      {/* Horizontal Bar Chart */}
      <div className="space-y-2">
        <div className="h-96 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 10, right: 30, left: 140, bottom: 10 }}
            >
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
              <XAxis
                type="number"
                tick={{ fontSize: 11, fill: '#64748b' }}
                label={{ value: 'Mean |SHAP Value| (Log-Odds Impact)', position: 'insideBottom', offset: -5, fontSize: 11, fill: '#64748b' }}
              />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fontSize: 11, fill: '#334155' }}
                interval={0}
                width={140}
              />
              <Tooltip
                formatter={(value: any) => [`${value}`, 'Mean |SHAP|']}
                contentStyle={{ borderRadius: '12px', border: '1px solid #e2e8f0', fontSize: '12px' }}
              />
              <Bar dataKey="mean_abs_shap" fill="#0d9488" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Ranked Summary Table */}
      <div className="overflow-x-auto pt-2">
        <table className="w-full text-xs text-left text-slate-600 border border-slate-200/80 rounded-xl overflow-hidden">
          <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200/80">
            <tr>
              <th className="px-3 py-2.5">Global Rank</th>
              <th className="px-3 py-2.5">Clinical Feature</th>
              <th className="px-3 py-2.5">Feature Type</th>
              <th className="px-3 py-2.5">Mean |SHAP Value|</th>
              <th className="px-3 py-2.5">Clinical Role</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {globalData.clinical_rankings.map((item) => (
              <tr key={item.feature} className="hover:bg-slate-50/60">
                <td className="px-3 py-2 font-mono font-bold text-slate-900">#{item.rank}</td>
                <td className="px-3 py-2 font-semibold text-slate-800">{item.display_name}</td>
                <td className="px-3 py-2 text-slate-500 uppercase text-[10px]">{item.type}</td>
                <td className="px-3 py-2 font-mono font-semibold text-teal-800">
                  {item.mean_abs_shap.toFixed(4)}
                </td>
                <td className="px-3 py-2 text-slate-500 text-[11px]">
                  {item.rank <= 3 ? (
                    <span className="font-semibold text-teal-700">Primary Risk Determinant</span>
                  ) : item.rank <= 7 ? (
                    <span>Secondary Factor</span>
                  ) : (
                    <span className="text-slate-400">Baseline Covariate</span>
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

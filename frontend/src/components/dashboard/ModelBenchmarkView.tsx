import React from 'react';
import type { ModelComparisonResponse } from '../../types/api';
import { Award, BarChart3, CheckCircle2, Info } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LabelList,
} from 'recharts';

interface ModelBenchmarkViewProps {
  comparison: ModelComparisonResponse | null;
  loading: boolean;
  error?: string | null;
}

// Short aliases for the grouped bar chart X-axis labels
const SHORT_NAMES: Record<string, string> = {
  'Logistic Regression': 'LR',
  'Random Forest': 'RF',
  'SVM': 'SVM',
  'KNN': 'KNN',
  'Gradient Boosting': 'GB',
};

// Color palette for the 5 models
const MODEL_COLORS: Record<string, string> = {
  'Logistic Regression': '#0d9488',
  'Random Forest': '#0284c7',
  'SVM': '#7c3aed',
  'KNN': '#ea580c',
  'Gradient Boosting': '#db2777',
};

const FULL_NAMES: Record<string, string> = {
  LR: 'Logistic Regression',
  RF: 'Random Forest',
  SVM: 'Support Vector Machine (SVM)',
  KNN: 'K-Nearest Neighbors (KNN)',
  GB: 'Gradient Boosting',
};

// Tooltip defined outside component to avoid recreating on each render
const BenchmarkTooltip = ({ active, payload, label }: any) => {
  if (active && payload?.length) {
    const fullName = FULL_NAMES[label] ?? label;
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-3 shadow-lg text-xs space-y-1 min-w-[180px]">
        <p className="font-bold text-slate-900 mb-1.5">{fullName}</p>
        {payload.map((p: any) => (
          <div key={p.name} className="flex items-center justify-between gap-4">
            <span style={{ color: p.color }} className="font-medium">{p.name}</span>
            <span className="font-mono font-bold text-slate-800">{p.value}%</span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

export const ModelBenchmarkView: React.FC<ModelBenchmarkViewProps> = ({
  comparison,
  loading,
  error,
}: ModelBenchmarkViewProps) => {
  if (loading) {
    return (
      <div className="bg-white rounded-2xl border border-slate-200/80 p-6 animate-pulse space-y-4">
        <div className="h-5 w-48 bg-slate-200 rounded"></div>
        <div className="h-64 bg-slate-100 rounded-xl"></div>
        <div className="h-48 bg-slate-100 rounded-xl"></div>
      </div>
    );
  }

  if (error || !comparison) {
    return (
      <div className="bg-white rounded-2xl border border-rose-200 p-6 text-rose-800 text-sm">
        Failed to load model comparison: {error || 'No benchmark data available'}
      </div>
    );
  }

  // Chart data — CV Accuracy vs Test Accuracy, keyed by short name
  const chartData = comparison.models.map((m) => ({
    name: SHORT_NAMES[m.model] ?? m.model,
    fullName: m.model,
    'CV Acc %': Number((m.cv_accuracy_mean * 100).toFixed(1)),
    'Test Acc %': Number((m.test_accuracy * 100).toFixed(1)),
    isChampion: m.model === comparison.champion_model,
  }));

  const championData = comparison.models.find((m) => m.model === comparison.champion_model);


  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-4">
        <div>
          <h3 className="text-base font-semibold text-slate-900 flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-teal-600" />
            Five-Algorithm ML Model Comparison
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Logistic Regression · Random Forest · SVM · KNN · Gradient Boosting — trained on UCI Cleveland N=303.
          </p>
        </div>
        <span className="text-xs font-mono bg-teal-50 text-teal-800 border border-teal-200 px-3 py-1 rounded-full font-medium self-start sm:self-auto whitespace-nowrap">
          Champion: {comparison.champion_model}
        </span>
      </div>

      {/* Small-test-set disclosure */}
      <div className="p-3 bg-amber-50 border border-amber-200/80 rounded-xl text-xs text-amber-800 flex items-start gap-2.5">
        <Info className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
        <p className="leading-relaxed">
          <strong>Important:</strong> The held-out test partition contains only <strong>61 records</strong> (20% of 303).
          Single percentage-point differences on this small sample may not be statistically meaningful.
          Cross-validation scores (training data only) are the primary model-selection metric.
        </p>
      </div>

      {/* Champion Highlight Banner */}
      {championData && (
        <div className="bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 text-white rounded-xl p-5 shadow-sm border border-slate-700/50 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="p-1 rounded-md bg-teal-500/20 text-teal-400">
                <Award className="w-4 h-4" />
              </span>
              <span className="text-xs font-semibold tracking-wider uppercase text-teal-400">Champion Model</span>
            </div>
            <h4 className="text-xl font-bold tracking-tight text-white">{championData.model}</h4>
            <p className="text-xs text-slate-400 max-w-xl">
              Live prediction champion: <strong>{championData.model}</strong>. All five configurations were finalized exclusively through training-only nested cross-validation, then evaluated once on the locked seed-2026 test partition for comparative reporting. These test results did not influence champion selection.
            </p>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 border-t md:border-t-0 md:border-l border-slate-700/80 pt-3 md:pt-0 md:pl-5">
            <div>
              <span className="text-[10px] font-semibold tracking-wider uppercase text-slate-400 block">CV F1 (Train)</span>
              <span className="text-lg font-bold font-mono text-teal-300">
                {(championData.cv_f1_mean ?? 0).toFixed(4)}
              </span>
              <span className="text-[10px] text-slate-400 block">Selection metric</span>
            </div>
            <div>
              <span className="text-[10px] font-semibold tracking-wider uppercase text-slate-400 block">CV Acc (Train)</span>
              <span className="text-lg font-bold font-mono text-white">
                {(championData.cv_accuracy_mean * 100).toFixed(1)}%
              </span>
              <span className="text-[10px] text-slate-400 block">±{(championData.cv_accuracy_std * 100).toFixed(1)}%</span>
            </div>
            <div>
              <span className="text-[10px] font-semibold tracking-wider uppercase text-slate-400 block">Test Acc (N=61)</span>
              <span className="text-lg font-bold font-mono text-white">
                {(championData.test_accuracy * 100).toFixed(1)}%
              </span>
              <span className="text-[10px] text-slate-400 block">Held-out, reported only</span>
            </div>
            <div>
              <span className="text-[10px] font-semibold tracking-wider uppercase text-slate-400 block">ROC-AUC</span>
              <span className="text-lg font-bold font-mono text-white">
                {championData.test_roc_auc ? (championData.test_roc_auc * 100).toFixed(1) + '%' : 'N/A'}
              </span>
              <span className="text-[10px] text-slate-400 block">Test partition</span>
            </div>
          </div>
        </div>
      )}

      {/* Grouped Bar Chart: CV Accuracy vs Test Accuracy */}
      <div className="space-y-2 pt-2">
        <div className="flex items-center justify-between">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
            CV Accuracy vs. Held-out Test Accuracy — All 5 Algorithms
          </h4>
          <span className="text-[10px] text-slate-400 font-mono">LR=Logistic Reg · RF=Random Forest · SVM · KNN · GB=Gradient Boosting</span>
        </div>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              margin={{ top: 18, right: 16, left: -10, bottom: 10 }}
              barGap={4}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 12, fill: '#475569', fontWeight: 600 }}
                interval={0}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                domain={[60, 100]}
                tick={{ fontSize: 11, fill: '#64748b' }}
                unit="%"
                axisLine={false}
                tickLine={false}
              />
              <Tooltip content={<BenchmarkTooltip />} />
              <Legend
                wrapperStyle={{ fontSize: '12px', paddingTop: '8px' }}
                formatter={(value) => (
                  <span className="text-slate-600 font-medium">{value}</span>
                )}
              />
              <Bar dataKey="CV Acc %" fill="#0d9488" radius={[4, 4, 0, 0]} maxBarSize={40}>
                <LabelList dataKey="CV Acc %" position="top" style={{ fontSize: '10px', fill: '#0d9488', fontWeight: 700 }} formatter={(v: any) => `${v}%`} />
              </Bar>
              <Bar dataKey="Test Acc %" fill="#0284c7" radius={[4, 4, 0, 0]} maxBarSize={40}>
                <LabelList dataKey="Test Acc %" position="top" style={{ fontSize: '10px', fill: '#0284c7', fontWeight: 700 }} formatter={(v: any) => `${v}%`} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="text-[11px] text-slate-400 text-center leading-relaxed">
          <strong>CV Accuracy</strong> = mean 5-fold cross-validation accuracy on training data (242 samples).&nbsp;
          <strong>Test Accuracy</strong> = All five configurations were finalized exclusively through training-only nested cross-validation, then evaluated once on the locked seed-2026 test partition for comparative reporting. These test results did not influence champion selection.
        </p>
      </div>

      {/* Full 5-Model Benchmark Table */}
      <div className="space-y-2 pt-2">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          Complete Performance Table (All 5 Algorithms · UCI Cleveland N=303)
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left text-slate-600 border border-slate-200/80 rounded-xl overflow-hidden">
            <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200/80">
              <tr>
                <th className="px-3 py-2.5">Algorithm</th>
                <th className="px-3 py-2.5 text-teal-700">CV Acc (mean ± std)</th>
                <th className="px-3 py-2.5 text-teal-700">CV F1 ★</th>
                <th className="px-3 py-2.5">Test Acc</th>
                <th className="px-3 py-2.5">Precision</th>
                <th className="px-3 py-2.5">Recall</th>
                <th className="px-3 py-2.5">F1</th>
                <th className="px-3 py-2.5">ROC-AUC</th>
                <th className="px-3 py-2.5">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {comparison.models.map((m) => {
                const isChamp = m.model === comparison.champion_model;
                const color = MODEL_COLORS[m.model] ?? '#475569';
                return (
                  <tr key={m.model} className={isChamp ? 'bg-teal-50/40 font-medium' : 'hover:bg-slate-50/60'}>
                    <td className="px-3 py-2.5 text-slate-900 font-semibold">
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
                        {isChamp && <CheckCircle2 className="w-3.5 h-3.5 text-teal-600 flex-shrink-0" />}
                        <span>{m.model}</span>
                      </div>
                    </td>
                    <td className="px-3 py-2.5 font-mono text-teal-800 font-semibold">
                      {(m.cv_accuracy_mean * 100).toFixed(1)}%
                      <span className="text-slate-400 font-normal"> ±{(m.cv_accuracy_std * 100).toFixed(1)}%</span>
                    </td>
                    <td className="px-3 py-2.5 font-mono text-teal-700 font-semibold">
                      {m.cv_f1_mean ? m.cv_f1_mean.toFixed(4) : 'N/A'}
                    </td>
                    <td className="px-3 py-2.5 font-mono font-semibold text-slate-900">
                      {(m.test_accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="px-3 py-2.5 font-mono">
                      {m.test_precision != null ? (m.test_precision * 100).toFixed(1) + '%' : '—'}
                    </td>
                    <td className="px-3 py-2.5 font-mono">
                      {m.test_recall != null ? (m.test_recall * 100).toFixed(1) + '%' : '—'}
                    </td>
                    <td className="px-3 py-2.5 font-mono font-semibold text-slate-900">
                      {m.test_f1.toFixed(4)}
                    </td>
                    <td className="px-3 py-2.5 font-mono">
                      {m.test_roc_auc ? m.test_roc_auc.toFixed(4) : 'N/A'}
                    </td>
                    <td className="px-3 py-2.5">
                      {isChamp ? (
                        <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-teal-100 text-teal-800">
                          Champion
                        </span>
                      ) : (
                        <span className="text-[10px] text-slate-400">Benchmarked</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="text-[11px] text-slate-400 leading-relaxed">
          ★ <strong>CV F1</strong> is the selection criterion (training folds only — test set not consulted). All test metrics were computed <em>after</em> final model selection and are reported for audit purposes only. Small N=61 test set means individual percentage differences are uncertain.
        </p>
      </div>
    </div>
  );
};

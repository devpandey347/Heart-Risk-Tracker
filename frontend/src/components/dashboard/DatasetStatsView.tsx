import React from 'react';
import type { DatasetStatsResponse } from '../../types/api';
import { Database, Users } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

interface DatasetStatsViewProps {
  stats: DatasetStatsResponse | null;
  loading: boolean;
  error?: string | null;
}

export const DatasetStatsView: React.FC<DatasetStatsViewProps> = ({ stats, loading, error }) => {
  if (loading) {
    return (
      <div className="bg-white rounded-2xl border border-slate-200/80 p-6 animate-pulse space-y-4">
        <div className="h-5 w-48 bg-slate-200 rounded"></div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-20 bg-slate-100 rounded-xl"></div>
          ))}
        </div>
      </div>
    );
  }

  if (error || !stats) {
    return (
      <div className="bg-white rounded-2xl border border-rose-200 p-6 text-rose-800 text-sm">
        Failed to load dataset statistics: {error || 'No data available'}
      </div>
    );
  }

  const classData = [
    {
      name: 'No Disease Detected (0)',
      count: stats.class_distribution.no_disease_count,
      pct: stats.class_distribution.no_disease_pct,
      color: '#10b981', // Emerald
    },
    {
      name: 'Disease Indicated (1)',
      count: stats.class_distribution.disease_present_count,
      pct: stats.class_distribution.disease_present_pct,
      color: '#0284c7', // Sky Blue
    },
  ];

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-4">
        <div>
          <h3 className="text-base font-semibold text-slate-900 flex items-center gap-2">
            <Database className="w-4 h-4 text-teal-600" />
            UCI Cleveland Dataset Distribution
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Verified ground-truth exploratory statistics from the 303-patient Cleveland cohort.
          </p>
        </div>
        <span className="text-xs font-mono bg-slate-100 text-slate-600 px-2.5 py-1 rounded-full font-medium self-start sm:self-auto">
          303 Patients • 13 Input Features
        </span>
      </div>

      {/* Summary KPI grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-slate-50/80 border border-slate-200/60 rounded-xl p-3.5">
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">Total Cohort</span>
          <span className="text-2xl font-bold font-mono text-slate-900">{stats.total_samples}</span>
          <span className="text-[11px] text-slate-500 block mt-0.5">Cardiac patients</span>
        </div>

        <div className="bg-emerald-50/50 border border-emerald-200/60 rounded-xl p-3.5">
          <span className="text-[11px] font-semibold text-emerald-700 uppercase tracking-wider block">Healthy / No Disease</span>
          <span className="text-2xl font-bold font-mono text-emerald-900">{stats.class_distribution.no_disease_count}</span>
          <span className="text-[11px] text-emerald-600 block mt-0.5">{stats.class_distribution.no_disease_pct}% prevalence</span>
        </div>

        <div className="bg-sky-50/50 border border-sky-200/60 rounded-xl p-3.5">
          <span className="text-[11px] font-semibold text-sky-700 uppercase tracking-wider block">Disease Present</span>
          <span className="text-2xl font-bold font-mono text-sky-900">{stats.class_distribution.disease_present_count}</span>
          <span className="text-[11px] text-sky-600 block mt-0.5">{stats.class_distribution.disease_present_pct}% prevalence</span>
        </div>

        <div className="bg-amber-50/50 border border-amber-200/60 rounded-xl p-3.5">
          <span className="text-[11px] font-semibold text-amber-700 uppercase tracking-wider block">Imputed Cells</span>
          <span className="text-2xl font-bold font-mono text-amber-900">
            {Object.values(stats.missing_values).reduce((a, b) => a + b, 0)}
          </span>
          <span className="text-[11px] text-amber-700 block mt-0.5">ca (4) • thal (2)</span>
        </div>
      </div>

      {/* Class distribution chart & info */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center pt-2">
        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={classData}
                cx="50%"
                cy="50%"
                innerRadius={55}
                outerRadius={80}
                paddingAngle={4}
                dataKey="count"
              >
                {classData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value: any, name: any) => [`${value} patients`, name]}
                contentStyle={{ borderRadius: '12px', border: '1px solid #e2e8f0', fontSize: '12px' }}
              />
              <Legend verticalAlign="bottom" height={36} wrapperStyle={{ fontSize: '12px' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="space-y-3 text-xs text-slate-600">
          <div className="p-3 bg-slate-50 border border-slate-200/60 rounded-xl space-y-1.5">
            <h5 className="font-semibold text-slate-900 text-xs flex items-center gap-1.5">
              <Users className="w-3.5 h-3.5 text-teal-600" />
              Cohort Demographics & Biomarkers
            </h5>
            <ul className="space-y-1 text-slate-600">
              <li>• <strong>Age:</strong> Mean {stats.numerical_features_summary.age?.mean} yrs (range {stats.numerical_features_summary.age?.min}–{stats.numerical_features_summary.age?.max})</li>
              <li>• <strong>Resting BP:</strong> Mean {stats.numerical_features_summary.trestbps?.mean} mm Hg (median {stats.numerical_features_summary.trestbps?.median})</li>
              <li>• <strong>Serum Cholesterol:</strong> Mean {stats.numerical_features_summary.chol?.mean} mg/dl (median {stats.numerical_features_summary.chol?.median})</li>
              <li>• <strong>Max Heart Rate:</strong> Mean {stats.numerical_features_summary.thalach?.mean} bpm (median {stats.numerical_features_summary.thalach?.median})</li>
            </ul>
          </div>
          <p className="text-[11px] text-slate-400">
            Note: All missing cells in fluoroscopy vessels (ca) and thallium defect (thal) are imputed automatically by median/mode inside the sklearn pipeline.
          </p>
        </div>
      </div>
    </div>
  );
};

import React from 'react';
import { ArrowRight, Stethoscope, Sparkles } from 'lucide-react';

interface QuickStartCTAProps {
  onOpenEstimation: () => void;
  onOpenExplainability: () => void;
}

export const QuickStartCTA: React.FC<QuickStartCTAProps> = ({
  onOpenEstimation,
  onOpenExplainability,
}) => {
  return (
    <div className="relative overflow-hidden bg-gradient-to-r from-teal-900 via-teal-800 to-slate-900 text-white rounded-2xl p-6 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-6 border border-teal-700/50">
      <div className="space-y-1.5 max-w-xl">
        <span className="text-[10px] font-bold uppercase tracking-widest text-teal-300 bg-teal-800/60 px-2 py-0.5 rounded border border-teal-600/40 inline-block">
          Interactive Clinical Risk Intake
        </span>
        <h3 className="text-xl font-bold tracking-tight text-white">
          Evaluate a Cardiovascular Patient Profile
        </h3>
        <p className="text-xs sm:text-sm text-teal-100/90 leading-relaxed">
          Input 13 clinical biomarkers to compute disease risk probability, clinical risk tier, and view patient-specific SHAP feature attributions.
        </p>
      </div>

      <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 flex-shrink-0">
        <button
          onClick={onOpenEstimation}
          className="inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl bg-white text-teal-900 hover:bg-teal-50 font-semibold text-sm transition-all shadow-sm focus:outline-none focus:ring-2 focus:ring-white/40"
        >
          <Stethoscope className="w-4 h-4 text-teal-700" />
          <span>Launch Risk Estimation</span>
          <ArrowRight className="w-4 h-4" />
        </button>

        <button
          onClick={onOpenExplainability}
          className="inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-teal-800/80 hover:bg-teal-800 text-teal-100 border border-teal-500/50 font-medium text-sm transition-all focus:outline-none"
        >
          <Sparkles className="w-4 h-4 text-teal-300" />
          <span>View SHAP Summary</span>
        </button>
      </div>
    </div>
  );
};

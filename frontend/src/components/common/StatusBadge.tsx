import React from 'react';
import { Activity, AlertCircle, CheckCircle2 } from 'lucide-react';

interface StatusBadgeProps {
  status: 'connected' | 'degraded' | 'offline' | 'loading';
  championModel?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, championModel }) => {
  if (status === 'loading') {
    return (
      <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-600 border border-slate-200">
        <Activity className="w-3 h-3 animate-spin text-slate-500" />
        <span>Connecting API...</span>
      </div>
    );
  }

  if (status === 'connected') {
    return (
      <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-50 text-emerald-800 border border-emerald-200">
        <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
        <span>Model Ready</span>
        {championModel && (
          <span className="text-emerald-700/80 font-normal border-l border-emerald-300 pl-1.5">
            {championModel}
          </span>
        )}
      </div>
    );
  }

  if (status === 'degraded') {
    return (
      <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-50 text-amber-800 border border-amber-200">
        <AlertCircle className="w-3.5 h-3.5 text-amber-600" />
        <span>Backend Degraded</span>
      </div>
    );
  }

  return (
    <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-rose-50 text-rose-800 border border-rose-200">
      <span className="w-2 h-2 rounded-full bg-rose-500"></span>
      <span>API Offline</span>
    </div>
  );
};

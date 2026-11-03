import React from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  badge?: string;
  accent?: 'default' | 'teal' | 'navy' | 'emerald' | 'amber';
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  icon,
  badge,
  accent = 'default',
}) => {
  const accentStyles = {
    default: 'border-slate-200/80 bg-white hover:border-slate-300',
    teal: 'border-teal-200/80 bg-teal-50/40 hover:border-teal-300',
    navy: 'border-slate-300 bg-slate-900 text-white hover:border-slate-700',
    emerald: 'border-emerald-200/80 bg-emerald-50/40 hover:border-emerald-300',
    amber: 'border-amber-200/80 bg-amber-50/40 hover:border-amber-300',
  }[accent];

  const isNavy = accent === 'navy';

  return (
    <div className={`p-5 rounded-2xl border transition-all duration-200 shadow-sm ${accentStyles}`}>
      <div className="flex items-center justify-between gap-2 mb-2">
        <span className={`text-xs font-medium uppercase tracking-wider ${isNavy ? 'text-slate-400' : 'text-slate-500'}`}>
          {title}
        </span>
        {icon && (
          <div className={`p-2 rounded-xl ${isNavy ? 'bg-slate-800 text-teal-400' : 'bg-slate-100/80 text-slate-600'}`}>
            {icon}
          </div>
        )}
      </div>

      <div className="flex items-baseline gap-2">
        <span className={`text-2xl sm:text-3xl font-bold tracking-tight font-mono ${isNavy ? 'text-white' : 'text-slate-900'}`}>
          {value}
        </span>
        {badge && (
          <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-teal-100 text-teal-800">
            {badge}
          </span>
        )}
      </div>

      {subtitle && (
        <p className={`mt-2 text-xs leading-normal ${isNavy ? 'text-slate-400' : 'text-slate-500'}`}>
          {subtitle}
        </p>
      )}
    </div>
  );
};

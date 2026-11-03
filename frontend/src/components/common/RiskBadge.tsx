import React from 'react';
import type { RiskCategory } from '../../types/api';
import { ShieldAlert, ShieldCheck, AlertTriangle } from 'lucide-react';

interface RiskBadgeProps {
  category: RiskCategory;
  probability?: number;
  size?: 'sm' | 'md' | 'lg';
}

export const RiskBadge: React.FC<RiskBadgeProps> = ({ category, probability, size = 'md' }) => {
  const getStyles = () => {
    switch (category) {
      case 'Low Risk':
        return {
          container: 'bg-emerald-50 text-emerald-800 border-emerald-200',
          icon: <ShieldCheck className="w-4 h-4 text-emerald-600" />,
          accent: 'text-emerald-700',
        };
      case 'Moderate Risk':
        return {
          container: 'bg-amber-50 text-amber-800 border-amber-200',
          icon: <AlertTriangle className="w-4 h-4 text-amber-600" />,
          accent: 'text-amber-700',
        };
      case 'High Risk':
      default:
        return {
          container: 'bg-rose-50 text-rose-800 border-rose-200',
          icon: <ShieldAlert className="w-4 h-4 text-rose-600" />,
          accent: 'text-rose-700',
        };
    }
  };

  const styles = getStyles();
  const sizeClasses = {
    sm: 'text-xs px-2.5 py-0.5',
    md: 'text-sm px-3 py-1',
    lg: 'text-base px-4 py-2 font-semibold',
  }[size];

  return (
    <span className={`inline-flex items-center gap-2 rounded-full border font-medium shadow-sm ${styles.container} ${sizeClasses}`}>
      {styles.icon}
      <span>{category}</span>
      {probability !== undefined && (
        <span className={`font-mono text-xs font-semibold ${styles.accent} bg-white/70 px-1.5 py-0.5 rounded-full border border-current/20`}>
          {(probability * 100).toFixed(1)}%
        </span>
      )}
    </span>
  );
};

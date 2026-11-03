import React from 'react';
import { HeartPulse, Menu, X, ShieldAlert, Database } from 'lucide-react';
import { StatusBadge } from '../common/StatusBadge';

interface NavbarProps {
  status: 'connected' | 'degraded' | 'offline' | 'loading';
  championModel?: string;
  mobileMenuOpen: boolean;
  onToggleMobileMenu: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  status,
  championModel,
  mobileMenuOpen,
  onToggleMobileMenu,
}) => {
  return (
    <header className="sticky top-0 z-30 bg-white/95 backdrop-blur-md border-b border-slate-200/80 shadow-xs h-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-full">
        <div className="flex items-center justify-between h-full gap-4">
          {/* Brand & Clinical Title */}
          <div className="flex items-center gap-3">
            <button
              onClick={onToggleMobileMenu}
              className="md:hidden p-2 -ml-2 text-slate-600 hover:text-slate-900 rounded-lg hover:bg-slate-100 focus:outline-none"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>

            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-teal-700 via-teal-600 to-emerald-600 flex items-center justify-center text-white shadow-sm ring-2 ring-teal-500/20 flex-shrink-0">
                <HeartPulse className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-bold text-slate-900 tracking-tight text-base sm:text-lg">
                    Heart Risk Tracker
                  </span>
                  <span className="hidden sm:inline-flex text-[10px] font-bold tracking-wider uppercase px-2 py-0.5 bg-teal-50 text-teal-800 border border-teal-200/80 rounded-md">
                    Cardiology CDSS
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 hidden sm:block">
                  Cardiovascular Risk Stratification & SHAP XAI Engine
                </p>
              </div>
            </div>
          </div>

          {/* Center Clinical Guidance Pill (Desktop) */}
          <div className="hidden xl:flex items-center gap-2 px-3.5 py-1 rounded-full bg-slate-100/80 text-slate-700 border border-slate-200 text-xs">
            <ShieldAlert className="w-3.5 h-3.5 text-amber-600 flex-shrink-0" />
            <span className="text-[11px] font-medium text-slate-600">
              Educational CDSS · Clinical Judgment Precedes Algorithm
            </span>
          </div>

          {/* Right Status & Cohort Indicators */}
          <div className="flex items-center gap-2.5">
            <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-700 border border-slate-200">
              <Database className="w-3 h-3 text-teal-700" />
              <span>Cleveland Cohort (N=303)</span>
            </div>
            <StatusBadge status={status} championModel={championModel} />
          </div>
        </div>
      </div>
    </header>
  );
};

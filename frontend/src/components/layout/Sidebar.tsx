import React from 'react';
import { LayoutDashboard, Stethoscope, Sparkles, Database } from 'lucide-react';

export type NavTab = 'overview' | 'estimation' | 'explainability';

interface SidebarProps {
  currentTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  mobileMenuOpen: boolean;
  onCloseMobile: () => void;
  hasPatientResult?: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onSelectTab,
  mobileMenuOpen,
  onCloseMobile,
  hasPatientResult,
}) => {
  const navItems = [
    {
      id: 'overview' as NavTab,
      label: 'Overview Dashboard',
      description: 'Model benchmarks & cohort data',
      icon: <LayoutDashboard className="w-4 h-4" />,
    },
    {
      id: 'estimation' as NavTab,
      label: 'Patient Risk Estimation',
      description: '13-biomarker clinical intake',
      icon: <Stethoscope className="w-4 h-4" />,
      badge: hasPatientResult ? 'Evaluated' : undefined,
    },
    {
      id: 'explainability' as NavTab,
      label: 'SHAP Diagnostic XAI',
      description: 'Patient waterfall & global impact',
      icon: <Sparkles className="w-4 h-4" />,
    },
  ];

  const handleNavClick = (tab: NavTab) => {
    onSelectTab(tab);
    onCloseMobile();
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {mobileMenuOpen && (
        <div
          className="fixed inset-0 bg-slate-900/50 z-40 md:hidden backdrop-blur-xs transition-opacity"
          onClick={onCloseMobile}
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed md:sticky top-16 z-20 h-[calc(100vh-4rem)] w-64 bg-white border-r border-slate-200/80 flex flex-col justify-between transition-transform duration-300 ease-in-out md:translate-x-0 ${
          mobileMenuOpen ? 'translate-x-0 !z-50 shadow-2xl' : '-translate-x-full'
        }`}
      >
        <div className="p-4 space-y-6 overflow-y-auto">
          {/* Main Navigation */}
          <div>
            <div className="px-3 mb-2.5 text-[10px] font-bold text-slate-400 uppercase tracking-widest">
              Clinical Protocols
            </div>
            <nav className="space-y-1">
              {navItems.map((item) => {
                const isActive = currentTab === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => handleNavClick(item.id)}
                    className={`w-full flex items-start gap-3 px-3 py-2.5 rounded-xl text-left transition-all ${
                      isActive
                        ? 'bg-teal-50/90 text-teal-950 font-semibold border border-teal-200/90 shadow-2xs'
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                    }`}
                  >
                    <div
                      className={`p-1.5 rounded-lg mt-0.5 ${
                        isActive ? 'bg-teal-700 text-white shadow-2xs' : 'bg-slate-100 text-slate-500'
                      }`}
                    >
                      {item.icon}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold truncate">{item.label}</span>
                        {item.badge && (
                          <span className="text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-200">
                            {item.badge}
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-slate-400 truncate font-normal mt-0.5">
                        {item.description}
                      </p>
                    </div>
                  </button>
                );
              })}
            </nav>
          </div>

          {/* Quick Registry Card */}
          <div className="bg-slate-50/80 border border-slate-200/80 rounded-xl p-3.5 space-y-2">
            <div className="flex items-center justify-between text-xs font-semibold text-slate-800">
              <span className="flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-teal-700" />
                <span>Cleveland Registry</span>
              </span>
              <span className="text-[10px] font-mono text-slate-500 bg-white px-1.5 py-0.5 rounded border border-slate-200">
                N=303
              </span>
            </div>
            <p className="text-[11px] text-slate-500 leading-relaxed">
              Standard clinical cardiac cohort with 13 physiological & diagnostic variables. Evaluated on locked test partition (seed 2026).
            </p>
          </div>
        </div>

        {/* Footer Specification Card */}
        <div className="p-3.5 border-t border-slate-200/80 bg-slate-50/60 text-xs">
          <div className="flex items-center justify-between text-slate-600 mb-1">
            <span className="font-semibold text-[11px]">FastAPI CDSS Core</span>
            <span className="font-mono text-[10px] text-teal-700 font-bold bg-teal-50 px-1.5 py-0.2 rounded border border-teal-200">
              v2.0.0
            </span>
          </div>
          <div className="text-[11px] text-slate-500 flex items-center justify-between">
            <span>Champion Model</span>
            <span className="font-semibold text-slate-700">SVM (Sigmoid)</span>
          </div>
          <div className="text-[10px] text-slate-400 mt-1">
            KernelExplainer XAI Engine
          </div>
        </div>
      </aside>
    </>
  );
};

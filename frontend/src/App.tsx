import { useEffect, useState } from 'react';
import type {
  DatasetStatsResponse,
  GlobalExplanationResponse,
  HealthResponse,
  ModelComparisonResponse,
  PatientExplanationResponse,
} from './types/api';
import { apiService } from './services/api';
import { Navbar } from './components/layout/Navbar';
import { Sidebar, type NavTab } from './components/layout/Sidebar';
import { DisclaimerBanner } from './components/common/DisclaimerBanner';
import { OverviewView } from './views/OverviewView';
import { EstimationView } from './views/EstimationView';
import { ExplainabilityView } from './views/ExplainabilityView';

export function App() {
  const [currentTab, setCurrentTab] = useState<NavTab>('overview');
  const [mobileMenuOpen, setMobileMenuOpen] = useState<boolean>(false);

  // Global API states
  const [healthStatus, setHealthStatus] = useState<'connected' | 'degraded' | 'offline' | 'loading'>('loading');
  const [healthData, setHealthData] = useState<HealthResponse | null>(null);
  const [comparisonData, setComparisonData] = useState<ModelComparisonResponse | null>(null);
  const [datasetStatsData, setDatasetStatsData] = useState<DatasetStatsResponse | null>(null);
  const [globalShapData, setGlobalShapData] = useState<GlobalExplanationResponse | null>(null);

  // Cross-view state (persists patient explanation when moving between tabs)
  const [patientExplanation, setPatientExplanation] = useState<PatientExplanationResponse | null>(null);

  const [loadingDashboard, setLoadingDashboard] = useState<boolean>(true);
  const [loadingGlobalShap, setLoadingGlobalShap] = useState<boolean>(true);
  const [globalShapError, setGlobalShapError] = useState<string | null>(null);

  useEffect(() => {
    async function loadInitialData() {
      // 1. Health check
      try {
        const health = await apiService.getHealth();
        setHealthData(health);
        setHealthStatus(health.model_ready ? 'connected' : 'degraded');
      } catch {
        setHealthStatus('offline');
      }

      // 2. Dashboard comparison & dataset stats
      try {
        const [comp, stats] = await Promise.all([
          apiService.getModelComparison(),
          apiService.getDatasetStats(),
        ]);
        setComparisonData(comp);
        setDatasetStatsData(stats);
      } catch (err: any) {
        console.error('Failed to load dashboard metrics:', err);
      } finally {
        setLoadingDashboard(false);
      }

      // 3. Global SHAP data
      try {
        const globalData = await apiService.getGlobalExplanation();
        setGlobalShapData(globalData);
      } catch (err: any) {
        setGlobalShapError(err.message || 'Failed to load global SHAP data');
      } finally {
        setLoadingGlobalShap(false);
      }
    }

    loadInitialData();
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans">
      {/* Top Navbar */}
      <Navbar
        status={healthStatus}
        championModel={healthData?.champion_model}
        mobileMenuOpen={mobileMenuOpen}
        onToggleMobileMenu={() => setMobileMenuOpen(!mobileMenuOpen)}
      />

      {/* Main Body with Sidebar & Content */}
      <div className="flex-1 flex max-w-7xl w-full mx-auto">
        <Sidebar
          currentTab={currentTab}
          onSelectTab={setCurrentTab}
          mobileMenuOpen={mobileMenuOpen}
          onCloseMobile={() => setMobileMenuOpen(false)}
          hasPatientResult={patientExplanation !== null}
        />

        {/* View Container */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 overflow-y-auto space-y-6">
          {/* Prominently placed, safely contained medical evaluation notice */}
          <DisclaimerBanner compact />

          {currentTab === 'overview' && (
            <OverviewView
              health={healthData}
              comparison={comparisonData}
              datasetStats={datasetStatsData}
              loading={loadingDashboard}
              onNavigateToEstimation={() => setCurrentTab('estimation')}
              onNavigateToExplainability={() => setCurrentTab('explainability')}
            />
          )}

          {currentTab === 'estimation' && (
            <EstimationView
              onExplanationGenerated={(exp) => setPatientExplanation(exp)}
            />
          )}

          {currentTab === 'explainability' && (
            <ExplainabilityView
              patientExplanation={patientExplanation}
              globalData={globalShapData}
              loadingGlobal={loadingGlobalShap}
              globalError={globalShapError}
              onNavigateToEstimation={() => setCurrentTab('estimation')}
            />
          )}
        </main>
      </div>

      {/* Global Footer */}
      <footer className="border-t border-slate-200/80 bg-white py-6 text-center text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-4 space-y-1">
          <p>
            Heart Risk Tracker • Educational Cardiovascular Health Analytics & SHAP Explainability
          </p>
          <p className="text-[11px] text-slate-400">
            Powered by FastAPI, scikit-learn & SHAP. This tool does not provide medical diagnoses.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;

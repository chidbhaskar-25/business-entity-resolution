import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import type { PageId } from './components/Sidebar';
import { Header } from './components/Header';
import { DashboardPage } from './pages/DashboardPage';
import { DatasetExplorerPage } from './pages/DatasetExplorerPage';
import { MatchExplorerPage } from './pages/MatchExplorerPage';
import { ExperimentsPage } from './pages/ExperimentsPage';
import { EvaluationPage } from './pages/EvaluationPage';
import { ValidationPage } from './pages/ValidationPage';
import { SubmissionPage } from './pages/SubmissionPage';
import { fetchDatasetStats, fetchHealth } from './api';
import type { DatasetStats, HealthStatus } from './types';

export const App: React.FC = () => {
  const [activePage, setActivePage] = useState<PageId>('dashboard');
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [datasetStats, setDatasetStats] = useState<DatasetStats | null>(null);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);

  const loadInitialData = async () => {
    setIsRefreshing(true);
    try {
      const [h, ds] = await Promise.all([
        fetchHealth().catch(() => null),
        fetchDatasetStats().catch(() => null),
      ]);
      if (h) setHealth(h);
      if (ds) setDatasetStats(ds);
    } catch (e) {
      console.warn('Initial sync warning:', e);
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    queueMicrotask(() => {
      loadInitialData();
    });
  }, []);

  const renderContent = () => {
    switch (activePage) {
      case 'dashboard':
        return (
          <DashboardPage
            health={health}
            datasetStats={datasetStats}
            onNavigate={(p) => setActivePage(p as PageId)}
          />
        );
      case 'dataset':
        return <DatasetExplorerPage stats={datasetStats} />;
      case 'matcher':
        return <MatchExplorerPage />;
      case 'evaluation':
        return <EvaluationPage />;
      case 'experiments':
        return <ExperimentsPage />;
      case 'validation':
        return <ValidationPage />;
      case 'submission':
        return <SubmissionPage />;
      default:
        return (
          <DashboardPage
            health={health}
            datasetStats={datasetStats}
            onNavigate={(p) => setActivePage(p as PageId)}
          />
        );
    }
  };

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: 'var(--bg-app)' }}>
      {/* Sidebar Navigation */}
      <Sidebar
        activePage={activePage}
        onSelectPage={setActivePage}
        status={health?.status ?? 'online'}
        activeModel={health?.active_model ?? 'HistGradientBoosting'}
      />

      {/* Main Content Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <Header
          activePage={activePage}
          threshold={health?.optimal_threshold ?? 0.88}
          activeModel={health?.active_model ?? 'HistGradientBoosting'}
          onRefresh={loadInitialData}
          isRefreshing={isRefreshing}
        />

        <main style={{ flex: 1, padding: '2rem', maxWidth: '1400px', width: '100%', margin: '0 auto' }}>
          {renderContent()}
        </main>
      </div>
    </div>
  );
};

export default App;

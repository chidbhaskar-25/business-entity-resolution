import React from 'react';
import { Target, Award, RefreshCw } from 'lucide-react';
import type { PageId } from './Sidebar';

interface HeaderProps {
  activePage: PageId;
  threshold: number;
  activeModel: string;
  onRefresh: () => void;
  isRefreshing: boolean;
}

const PAGE_TITLES: Record<PageId, { title: string; subtitle: string }> = {
  dashboard: {
    title: 'Resolution Command Center',
    subtitle: 'Real-time metrics, pipeline telemetry, and performance benchmarks',
  },
  dataset: {
    title: 'Dataset Profile & Distributions',
    subtitle: 'Source 1-3 schemas, country partitions, and ground truth characteristics',
  },
  matcher: {
    title: 'Match & Candidate Explorer',
    subtitle: 'Trace candidate blocking, pairwise features, model probabilities, and decision gating',
  },
  evaluation: {
    title: 'Macro F0.5 Evaluation & Threshold Optimizer',
    subtitle: 'Precision-weighted metric analysis, threshold sweep curve, and singleton preservation',
  },
  experiments: {
    title: 'Model & Feature Experiments',
    subtitle: 'Controlled benchmarks from baseline to winning HistGradientBoosting model',
  },
  validation: {
    title: 'Strict Submission Validator',
    subtitle: 'Automated 11-rule submission compliance and constraint verification',
  },
  submission: {
    title: 'Pipeline Inference & Submission Generator',
    subtitle: 'Batch scoring, singleton preservation, and competition TSV production',
  },
};

export const Header: React.FC<HeaderProps> = ({
  activePage,
  threshold,
  activeModel,
  onRefresh,
  isRefreshing,
}) => {
  const meta = PAGE_TITLES[activePage];

  return (
    <header
      style={{
        padding: '1.25rem 2rem',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        backgroundColor: 'rgba(13, 18, 29, 0.65)',
        backdropFilter: 'blur(16px)',
        position: 'sticky',
        top: 0,
        zIndex: 40,
      }}
    >
      <div>
        <h1 style={{ fontSize: '1.45rem', fontWeight: 800, color: 'var(--text-primary)' }}>
          {meta.title}
        </h1>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>
          {meta.subtitle}
        </p>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
        {/* Metric Chip */}
        <div className="badge badge-emerald" style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}>
          <Award size={14} />
          <span>Macro F0.5: <strong>0.9782</strong></span>
        </div>

        {/* Threshold Chip */}
        <div className="badge badge-indigo" style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}>
          <Target size={14} />
          <span>Threshold: <strong>{threshold.toFixed(2)}</strong></span>
        </div>

        {/* Model Chip */}
        <div className="badge badge-cyan" style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}>
          <span>Model: <strong>{activeModel.includes('hist') ? 'HistGradBoost' : 'LogReg'}</strong></span>
        </div>

        {/* Refresh Button */}
        <button
          onClick={onRefresh}
          className="btn btn-secondary"
          style={{ padding: '0.45rem 0.75rem', fontSize: '0.8rem' }}
          title="Refresh Data from API"
          disabled={isRefreshing}
        >
          <RefreshCw size={14} className={isRefreshing ? 'animate-spin' : ''} />
          <span>Sync</span>
        </button>
      </div>
    </header>
  );
};

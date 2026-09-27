import React, { useState, useEffect } from 'react';
import { FlaskConical, Award, Zap, Layers } from 'lucide-react';
import { fetchExperiments } from '../api';
import type { ExperimentItem } from '../types';

export const ExperimentsPage: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    fetchExperiments()
      .then((data) => setExperiments(data.experiments))
      .catch((err) => console.error('Failed to load experiments', err))
      .finally(() => setIsLoading(false));
  }, []);

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Experiments Banner */}
      <div className="glass-panel" style={{ padding: '1.5rem 2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.35rem' }}>
              <FlaskConical size={22} color="var(--cyan)" />
              <h2 style={{ fontSize: '1.35rem', color: '#ffffff', fontWeight: 800 }}>
                Controlled Experimentation Matrix (A through F)
              </h2>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', maxWidth: '800px', lineHeight: 1.6 }}>
              Rigorous ablation study isolating feature groups, blocking capacities, and model architectures.
              Improvements selected strictly based on validation Macro F0.5 evidence.
            </p>
          </div>

          <div className="badge badge-emerald" style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}>
            <Award size={16} />
            <span>Winning Model: Exp E (Macro F0.5: 0.9782)</span>
          </div>
        </div>
      </div>

      {/* Experiments Comparison Table */}
      <div className="glass-panel" style={{ overflow: 'hidden' }}>
        <div style={{ padding: '1.25rem 1.5rem', borderBottom: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h3 style={{ fontSize: '1.05rem', color: 'var(--text-primary)', fontWeight: 700 }}>
            Model & Feature Ablation Results
          </h3>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {isLoading ? 'Loading...' : `${experiments.length} Total Experiments`}
          </span>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
            <thead>
              <tr style={{ backgroundColor: 'rgba(255, 255, 255, 0.02)', borderBottom: '1px solid var(--border-subtle)' }}>
                <th style={{ padding: '0.9rem 1.25rem', color: 'var(--text-muted)', fontWeight: 600 }}>Exp</th>
                <th style={{ padding: '0.9rem 1.25rem', color: 'var(--text-muted)', fontWeight: 600 }}>Model Architecture</th>
                <th style={{ padding: '0.9rem 1.25rem', color: 'var(--text-muted)', fontWeight: 600 }}>Feature Configuration</th>
                <th style={{ padding: '0.9rem 1.25rem', color: 'var(--text-muted)', fontWeight: 600 }}>Blocking</th>
                <th style={{ padding: '0.9rem 1.25rem', color: 'var(--text-muted)', fontWeight: 600 }}>Threshold</th>
                <th style={{ padding: '0.9rem 1.25rem', color: 'var(--text-muted)', fontWeight: 600 }}>Cand Recall</th>
                <th style={{ padding: '0.9rem 1.25rem', color: 'var(--text-muted)', fontWeight: 600 }}>Precision</th>
                <th style={{ padding: '0.9rem 1.25rem', color: 'var(--text-muted)', fontWeight: 600 }}>Recall</th>
                <th style={{ padding: '0.9rem 1.25rem', color: 'var(--text-muted)', fontWeight: 700 }}>Macro F0.5</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((exp) => {
                const isWinner = exp.experiment_id === 'E';
                return (
                  <tr
                    key={exp.experiment_id}
                    style={{
                      borderBottom: '1px solid var(--border-subtle)',
                      backgroundColor: isWinner ? 'rgba(16, 185, 129, 0.08)' : 'transparent',
                      transition: 'background-color 0.15s ease',
                    }}
                  >
                    <td style={{ padding: '0.9rem 1.25rem', fontFamily: 'var(--font-mono)', fontWeight: 800, color: isWinner ? 'var(--emerald)' : 'var(--cyan)' }}>
                      {exp.experiment_id} {isWinner && '★'}
                    </td>
                    <td style={{ padding: '0.9rem 1.25rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                      {exp.model}
                    </td>
                    <td style={{ padding: '0.9rem 1.25rem', color: 'var(--text-secondary)' }}>
                      {exp.features}
                    </td>
                    <td style={{ padding: '0.9rem 1.25rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                      {exp.blocking}
                    </td>
                    <td style={{ padding: '0.9rem 1.25rem', fontFamily: 'var(--font-mono)', color: 'var(--amber)', fontWeight: 700 }}>
                      τ = {exp.threshold?.toFixed(2) ?? '-'}
                    </td>
                    <td style={{ padding: '0.9rem 1.25rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                      {exp.candidate_recall ? `${(exp.candidate_recall * 100).toFixed(1)}%` : '-'}
                    </td>
                    <td style={{ padding: '0.9rem 1.25rem', fontFamily: 'var(--font-mono)', color: 'var(--cyan)' }}>
                      {exp.precision ? `${(exp.precision * 100).toFixed(2)}%` : '-'}
                    </td>
                    <td style={{ padding: '0.9rem 1.25rem', fontFamily: 'var(--font-mono)', color: 'var(--primary)' }}>
                      {exp.recall ? `${(exp.recall * 100).toFixed(2)}%` : '-'}
                    </td>
                    <td style={{ padding: '0.9rem 1.25rem', fontFamily: 'var(--font-mono)', fontSize: '1rem', fontWeight: 800, color: isWinner ? 'var(--emerald)' : 'var(--text-primary)' }}>
                      {exp.f05 ? exp.f05.toFixed(4) : '-'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Key Empirical Insights Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.25rem' }}>
        <div className="glass-card" style={{ borderLeft: '4px solid var(--emerald)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.6rem' }}>
            <Award size={18} color="var(--emerald)" />
            <h4 style={{ fontSize: '1rem', color: 'var(--text-primary)', fontWeight: 700 }}>
              Why Gradient Boosting Won (Exp E)
            </h4>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            Tree-based boosting captures non-linear feature gating: address similarity is heavily downweighted if
            country does not match, and door number mismatch is penalized non-linearly. Macro F0.5 reached <strong>0.9782</strong> (+0.0145 over baseline).
          </p>
        </div>

        <div className="glass-card" style={{ borderLeft: '4px solid var(--amber)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.6rem' }}>
            <Zap size={18} color="var(--amber)" />
            <h4 style={{ fontSize: '1rem', color: 'var(--text-primary)', fontWeight: 700 }}>
              The High-Threshold Finding (τ = 0.88)
            </h4>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            Because F0.5 weights precision 4x higher than recall, standard 0.50 decision thresholds generate false positive matches
            on borderline candidates, collapsing singleton accuracy. Sweeping thresholds to τ = 0.88 boosted singleton accuracy from 81.6% to <strong>97.4%</strong>.
          </p>
        </div>

        <div className="glass-card" style={{ borderLeft: '4px solid var(--cyan)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.6rem' }}>
            <Layers size={18} color="var(--cyan)" />
            <h4 style={{ fontSize: '1rem', color: 'var(--text-primary)', fontWeight: 700 }}>
              Blocking Capacity Scaling (Exp F)
            </h4>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            Expanding blocking to $k=35$ boosted candidate recall from 92.1% to <strong>96.02%</strong>, achieving near-perfect candidate coverage.
            Standard $k=25$ provides the ideal trade-off between inference speed and recall.
          </p>
        </div>
      </div>
    </div>
  );
};

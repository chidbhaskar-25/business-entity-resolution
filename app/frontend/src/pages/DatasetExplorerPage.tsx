import React from 'react';
import { Database, Globe2, AlertTriangle, FileSpreadsheet, CheckCircle2, ShieldAlert } from 'lucide-react';
import type { DatasetStats } from '../types';

interface DatasetExplorerProps {
  stats?: DatasetStats | null;
}

export const DatasetExplorerPage: React.FC<DatasetExplorerProps> = ({ stats: _stats }) => {
  const trainFiles = [
    { name: 'train_source1.tsv', desc: 'Deduplicated Reference Source', rows: '5,000 entities', size: '1.2 MB', prefix: 'S1-' },
    { name: 'train_source2.tsv', desc: 'Noisy Listing Source 2', rows: '25,000 entities', size: '6.4 MB', prefix: 'S2-' },
    { name: 'train_source3.tsv', desc: 'Noisy Registry Source 3', rows: '25,000 entities', size: '6.1 MB', prefix: 'S3-' },
    { name: 'train_ground_truth.tsv', desc: 'True Match Links Mapping', rows: '15,888 true links', size: '420 KB', prefix: 'S1- -> S2-/S3-' },
  ];

  const testFiles = [
    { name: 'test_source1.tsv', desc: 'Target Entities to Resolve', rows: '1,732,545 entities', size: '175.0 MB', prefix: 'S1-' },
    { name: 'test_source2.tsv', desc: 'Unlabeled Candidate Source 2', rows: '4,984,795 entities', size: '509.5 MB', prefix: 'S2-' },
    { name: 'test_source3.tsv', desc: 'Unlabeled Candidate Source 3', rows: '4,984,794 entities', size: '506.0 MB', prefix: 'S3-' },
  ];

  const countries = [
    { name: 'United States (US)', share: 45, count: '~780k', split: 'Train & Test' },
    { name: 'India (India)', share: 40, count: '~690k', split: 'Train & Test' },
    { name: 'France (France)', share: 15, count: '~260k', split: 'Test Set Only (Open-Set)' },
  ];

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Intro Header */}
      <div className="glass-panel" style={{ padding: '1.5rem 2rem' }}>
        <h2 style={{ fontSize: '1.4rem', color: '#ffffff', marginBottom: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <Database size={22} color="var(--cyan)" />
          <span>Multi-Source Dataset Specifications</span>
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: 1.6 }}>
          The competition dataset comprises 3 distinct sources with varying degrees of noise, address completeness,
          and naming conventions. Test evaluations introduce open-set country partitions (France) requiring strict
          country-isolated blocking.
        </p>
      </div>

      {/* Train and Test Files Comparison */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '1.5rem' }}>
        {/* Train Files Card */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
            <h3 style={{ fontSize: '1.1rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <FileSpreadsheet size={18} color="var(--emerald)" />
              <span>Training Dataset (data/train)</span>
            </h3>
            <span className="badge badge-emerald">Ground Truth Available</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            {trainFiles.map((file, idx) => (
              <div
                key={idx}
                style={{
                  backgroundColor: 'rgba(255, 255, 255, 0.03)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-md)',
                  padding: '0.85rem 1rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                }}
              >
                <div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                    {file.name}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{file.desc}</div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-secondary)' }}>{file.rows}</div>
                  <span className="badge badge-indigo" style={{ fontSize: '0.65rem' }}>{file.prefix}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Test Files Card */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
            <h3 style={{ fontSize: '1.1rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <FileSpreadsheet size={18} color="var(--cyan)" />
              <span>Test Dataset (data/test)</span>
            </h3>
            <span className="badge badge-cyan">~11.7 Million Total Rows</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            {testFiles.map((file, idx) => (
              <div
                key={idx}
                style={{
                  backgroundColor: 'rgba(255, 255, 255, 0.03)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-md)',
                  padding: '0.85rem 1rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                }}
              >
                <div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                    {file.name}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{file.desc}</div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-secondary)' }}>{file.rows}</div>
                  <span className="badge badge-cyan" style={{ fontSize: '0.65rem' }}>{file.size}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Country Distribution & Open-Set Partitions */}
      <div className="glass-panel" style={{ padding: '1.5rem 2rem' }}>
        <h3 style={{ fontSize: '1.1rem', color: 'var(--text-primary)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Globe2 size={18} color="var(--primary)" />
          <span>Country Partitions & Open-Set Generalization</span>
        </h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.25rem' }}>
          Entities never match across national jurisdictions. Blocking automatically creates isolated country inverted indexes.
          France is an open-set label in the test set.
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {countries.map((c, idx) => (
            <div key={idx} style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600 }}>
                <span style={{ color: 'var(--text-primary)' }}>{c.name}</span>
                <span style={{ color: 'var(--text-muted)' }}>{c.count} ({c.share}%) • <strong style={{ color: 'var(--cyan)' }}>{c.split}</strong></span>
              </div>
              <div style={{ height: '8px', width: '100%', backgroundColor: 'rgba(255, 255, 255, 0.08)', borderRadius: '4px', overflow: 'hidden' }}>
                <div
                  style={{
                    height: '100%',
                    width: `${c.share}%`,
                    background: idx === 2 ? 'linear-gradient(90deg, #ec4899, #f43f5e)' : 'linear-gradient(90deg, #6366f1, #06b6d4)',
                    borderRadius: '4px',
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Ground Truth & Quality Checks */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem' }}>
        <div className="glass-card" style={{ borderLeft: '3px solid var(--amber)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
            <AlertTriangle size={18} color="var(--amber)" />
            <h4 style={{ fontSize: '0.95rem', color: 'var(--text-primary)' }}>Missing Address Rates</h4>
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            ~3.1% of records in Source 2 and Source 3 omit business addresses. Our feature builder incorporates
            <code>addr_is_missing_either</code> indicator features to prevent false penalty.
          </p>
        </div>

        <div className="glass-card" style={{ borderLeft: '3px solid var(--emerald)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
            <CheckCircle2 size={18} color="var(--emerald)" />
            <h4 style={{ fontSize: '0.95rem', color: 'var(--text-primary)' }}>Singleton Preservation</h4>
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            5.58% of entities in training have zero matches. Any false link on a singleton yields 0.0 F0.5.
            Our calibrated threshold ($\tau = 0.88$) achieves 97.4% accuracy on singletons.
          </p>
        </div>

        <div className="glass-card" style={{ borderLeft: '3px solid var(--primary)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
            <ShieldAlert size={18} color="var(--primary)" />
            <h4 style={{ fontSize: '0.95rem', color: 'var(--text-primary)' }}>Multi-Match Resolution</h4>
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            94.42% of non-singleton entities match multiple target records across both Source 2 and Source 3.
            Matches are scored independently and deduplicated.
          </p>
        </div>
      </div>
    </div>
  );
};

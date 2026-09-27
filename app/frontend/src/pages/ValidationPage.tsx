import React, { useState, useEffect } from 'react';
import { ShieldCheck, CheckCircle2, XCircle, RefreshCw, FileText } from 'lucide-react';
import { triggerValidation } from '../api';
import type { ValidationReport } from '../types';

export const ValidationPage: React.FC = () => {
  const [report, setReport] = useState<ValidationReport | null>(null);
  const [isValidating, setIsValidating] = useState<boolean>(false);
  const [checkTargetIds, setCheckTargetIds] = useState<boolean>(false);
  const [targetSet, setTargetSet] = useState<'official' | 'sampled'>('official');
  const [sampleSize, setSampleSize] = useState<number>(20);

  const runValidation = async (mode: 'official' | 'sampled' = targetSet) => {
    setIsValidating(true);
    try {
      const isOfficial = mode === 'official';
      const res = await triggerValidation({
        matchingPath: isOfficial ? 'output/matching_results.tsv' : 'outputs/matching_results.tsv',
        candidatePath: isOfficial ? 'output/candidate_pairs.tsv' : 'outputs/candidate_pairs.tsv',
        checkTargetIds,
        s1SampleSize: isOfficial ? null : (sampleSize > 0 ? sampleSize : null),
      });
      setReport(res);
    } catch (err: any) {
      console.error('Validation failed', err);
    } finally {
      setIsValidating(false);
    }
  };

  useEffect(() => {
    queueMicrotask(() => {
      runValidation('official');
    });
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const criteriaLabels = [
    { key: 'correct_tsv_format', label: '1. Correct TSV format', desc: 'Strictly tab-delimited (.tsv), no CSV comma corruption' },
    { key: 'exact_column_names', label: '2. Exact column headers', desc: 'source1_entity_id\\tmatched_entity_ids / candidate_entity_ids' },
    { key: 'every_s1_test_id_exists', label: '3. Every Source 1 test entity present', desc: 'Zero missing entities from test_source1.tsv' },
    { key: 'exactly_one_row_per_s1', label: '4. Exactly one row per Source 1 entity', desc: 'Total row count strictly equals test entity count' },
    { key: 'no_duplicate_s1_rows', label: '5. No duplicate Source 1 rows', desc: 'Each S1 ID appears on exactly one line' },
    { key: 'only_s2_s3_matches', label: '6. Only S2/S3 IDs appear as matches', desc: 'Zero self-matches (no S1 IDs in match list)' },
    { key: 'all_ids_exist_in_test_data', label: '7. All entity IDs exist in test data', desc: 'Validates IDs against Source 1, 2, and 3 test sets' },
    { key: 'no_duplicate_ids_in_lists', label: '8. No duplicate IDs inside lists', desc: 'No intra-list repeated IDs (e.g. S2-1,S2-1)' },
    { key: 'candidate_ids_valid', label: '9. Candidate IDs valid and compliant', desc: 'All candidates have S2-/S3- prefix and valid ID syntax' },
    { key: 'matches_subset_of_candidates', label: '10. Matches strictly a subset of candidates', desc: 'Every final link exists in candidate_pairs.tsv' },
    { key: 'empty_lists_correctly_represented', label: '11. Empty lists (singletons) correctly formatted', desc: 'Singletons formatted as empty string after tab: s1_id\\t\\n' },
  ];

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Top Banner & Trigger Controls */}
      <div className="glass-panel" style={{ padding: '1.5rem 2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1.25rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.35rem' }}>
              <ShieldCheck size={22} color="var(--emerald)" />
              <h2 style={{ fontSize: '1.35rem', color: '#ffffff', fontWeight: 800 }}>
                Strict Competition Submission Validator
              </h2>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', maxWidth: '750px', lineHeight: 1.6 }}>
              Verifies matching_results.tsv and candidate_pairs.tsv against every formatting rule,
              cardinality constraint, and schema rule required by the leaderboard scorer.
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem', flexWrap: 'wrap' }}>
            {/* Mode Switcher */}
            <div style={{ display: 'flex', backgroundColor: 'rgba(255, 255, 255, 0.05)', borderRadius: 'var(--radius-md)', padding: '0.2rem', border: '1px solid var(--border-subtle)' }}>
              <button
                onClick={() => {
                  setTargetSet('official');
                  runValidation('official');
                }}
                style={{
                  padding: '0.4rem 0.8rem',
                  fontSize: '0.78rem',
                  fontWeight: 600,
                  borderRadius: 'var(--radius-sm)',
                  border: 'none',
                  cursor: 'pointer',
                  backgroundColor: targetSet === 'official' ? 'var(--emerald)' : 'transparent',
                  color: targetSet === 'official' ? '#ffffff' : 'var(--text-secondary)',
                  transition: 'all 0.15s ease',
                }}
              >
                Official Package (output/)
              </button>
              <button
                onClick={() => {
                  setTargetSet('sampled');
                  runValidation('sampled');
                }}
                style={{
                  padding: '0.4rem 0.8rem',
                  fontSize: '0.78rem',
                  fontWeight: 600,
                  borderRadius: 'var(--radius-sm)',
                  border: 'none',
                  cursor: 'pointer',
                  backgroundColor: targetSet === 'sampled' ? 'var(--primary)' : 'transparent',
                  color: targetSet === 'sampled' ? '#ffffff' : 'var(--text-secondary)',
                  transition: 'all 0.15s ease',
                }}
              >
                Sample Runs (outputs/)
              </button>
            </div>

            {targetSet === 'sampled' && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                <span>Sample Size:</span>
                <input
                  type="number"
                  value={sampleSize}
                  onChange={(e) => setSampleSize(Number(e.target.value))}
                  min={0}
                  max={5000}
                  step={10}
                  style={{
                    width: '65px',
                    backgroundColor: 'rgba(255, 255, 255, 0.05)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius-sm)',
                    color: '#ffffff',
                    padding: '0.2rem 0.4rem',
                    fontSize: '0.8rem',
                  }}
                />
              </div>
            )}

            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={checkTargetIds}
                onChange={(e) => setCheckTargetIds(e.target.checked)}
                style={{ accentColor: 'var(--primary)' }}
              />
              <span>Verify 10M Target IDs</span>
            </label>

            <button
              onClick={() => runValidation(targetSet)}
              className="btn btn-emerald"
              disabled={isValidating}
              style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}
            >
              <RefreshCw size={15} className={isValidating ? 'animate-spin' : ''} />
              <span>{isValidating ? 'Validating...' : 'Run Strict Validation'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Result Status Banner */}
      {report && (
        <div
          className="glass-card"
          style={{
            borderLeft: report.is_valid ? '5px solid var(--emerald)' : '5px solid var(--rose)',
            backgroundColor: report.is_valid ? 'rgba(16, 185, 129, 0.08)' : 'rgba(244, 63, 94, 0.08)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '1.25rem 1.75rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            {report.is_valid ? (
              <CheckCircle2 size={26} color="var(--emerald)" />
            ) : (
              <XCircle size={26} color="var(--rose)" />
            )}
            <div>
              <div style={{ fontSize: '1.1rem', fontWeight: 800, color: '#ffffff' }}>
                {report.is_valid ? 'VALIDATION PASSED (100% Compliant)' : 'VALIDATION ISSUES DETECTED'}
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                {report.is_valid
                  ? 'All 11 submission criteria satisfied. Safe to package for leaderboard submission.'
                  : `${report.errors.length} formatting errors detected.`}
              </div>
            </div>
          </div>

          <span className={report.is_valid ? 'badge badge-emerald' : 'badge badge-rose'} style={{ fontSize: '0.85rem', padding: '0.4rem 0.85rem' }}>
            {report.status}
          </span>
        </div>
      )}

      {/* 11 Rules Checklist Grid */}
      <div className="glass-panel" style={{ padding: '1.5rem 2rem' }}>
        <h3 style={{ fontSize: '1.1rem', color: 'var(--text-primary)', marginBottom: '1.25rem' }}>
          Official Submission Criteria Checklist (11 Rules)
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '0.85rem' }}>
          {criteriaLabels.map((rule) => {
            const isPassed = report?.criteria_checks[rule.key] ?? false;
            return (
              <div
                key={rule.key}
                style={{
                  backgroundColor: 'rgba(255, 255, 255, 0.02)',
                  border: isPassed ? '1px solid rgba(16, 185, 129, 0.25)' : '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-md)',
                  padding: '0.9rem 1.1rem',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '0.75rem',
                }}
              >
                {isPassed ? (
                  <CheckCircle2 size={18} color="var(--emerald)" style={{ marginTop: '0.15rem', flexShrink: 0 }} />
                ) : (
                  <XCircle size={18} color="var(--rose)" style={{ marginTop: '0.15rem', flexShrink: 0 }} />
                )}
                <div>
                  <div style={{ fontSize: '0.875rem', fontWeight: 700, color: isPassed ? '#ffffff' : 'var(--rose)' }}>
                    {rule.label}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                    {rule.desc}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Raw Report Summary */}
      {report?.report_summary && (
        <div className="glass-panel" style={{ padding: '1.5rem 2rem' }}>
          <h3 style={{ fontSize: '1.05rem', color: 'var(--text-primary)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <FileText size={18} color="var(--cyan)" />
            <span>Detailed Validator Output Log</span>
          </h3>
          <pre
            style={{
              backgroundColor: 'rgba(0, 0, 0, 0.45)',
              padding: '1.25rem',
              borderRadius: 'var(--radius-md)',
              fontSize: '0.8rem',
              color: '#a5b4fc',
              overflowX: 'auto',
              border: '1px solid var(--border-subtle)',
              lineHeight: 1.5,
            }}
          >
            {report.report_summary}
          </pre>
        </div>
      )}
    </div>
  );
};

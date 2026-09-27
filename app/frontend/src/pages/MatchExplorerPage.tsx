import React, { useState, useEffect } from 'react';
import {
  Search,
  GitCompare,
  Building2,
  MapPin,
  Globe,
  CheckCircle2,
  XCircle,
  Sparkles,
} from 'lucide-react';
import { fetchEntityMatches } from '../api';
import type { EntityMatchResult } from '../types';

const SAMPLE_ENTITIES = [
  { id: 'S1-921369899', label: 'Match to S2 Listing', desc: 'Resolves to S2-699224357 with high address alignment' },
  { id: 'S1-778640743', label: 'Match to S3 Registry', desc: 'Resolves to S3-467616579 commercial registry entry' },
  { id: 'S1-714132312', label: 'Singleton Entity (0 Matches)', desc: 'Preserves compliant empty match list (s1\\t\\n)' },
  { id: 'S1-156285671', label: 'Singleton Entity (Candidates Filtered)', desc: 'Candidates rejected below decision threshold' },
];

export const MatchExplorerPage: React.FC = () => {
  const [selectedId, setSelectedId] = useState<string>('S1-921369899');
  const [searchInput, setSearchInput] = useState<string>('S1-921369899');
  const [entityData, setEntityData] = useState<EntityMatchResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const loadEntity = async (id: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await fetchEntityMatches(id);
      setEntityData(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load entity');
      setEntityData(null);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    queueMicrotask(() => {
      loadEntity(selectedId);
    });
  }, [selectedId]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchInput.trim()) {
      setSelectedId(searchInput.trim());
    }
  };

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Search & Preset Selection Bar */}
      <div className="glass-panel" style={{ padding: '1.5rem 2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <h2 style={{ fontSize: '1.25rem', color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Search size={20} color="var(--cyan)" />
              <span>Select Source 1 Entity to Trace</span>
            </h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginTop: '0.2rem' }}>
              Inspect candidate generation, feature vectors, model inference probabilities, and decision gating.
            </p>
          </div>

          <form onSubmit={handleSearch} style={{ display: 'flex', gap: '0.5rem', width: '100%', maxWidth: '420px' }}>
            <input
              type="text"
              className="input-field"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              placeholder="Enter Source 1 Entity ID (e.g. S1-921369899)..."
              style={{ fontFamily: 'var(--font-mono)' }}
            />
            <button type="submit" className="btn btn-primary" style={{ padding: '0.65rem 1.1rem' }}>
              Trace
            </button>
          </form>
        </div>

        {/* Preset Chips */}
        <div style={{ display: 'flex', gap: '0.65rem', flexWrap: 'wrap' }}>
          {SAMPLE_ENTITIES.map((item) => (
            <button
              key={item.id}
              onClick={() => {
                setSearchInput(item.id);
                setSelectedId(item.id);
              }}
              style={{
                backgroundColor: selectedId === item.id ? 'rgba(99, 102, 241, 0.25)' : 'rgba(255, 255, 255, 0.04)',
                border: selectedId === item.id ? '1px solid var(--primary)' : '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                padding: '0.5rem 0.85rem',
                color: selectedId === item.id ? '#ffffff' : 'var(--text-secondary)',
                fontSize: '0.8rem',
                cursor: 'pointer',
                textAlign: 'left',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                transition: 'all 0.15s ease',
              }}
            >
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--cyan)' }}>{item.id}</span>
              <span>•</span>
              <span>{item.label}</span>
            </button>
          ))}
        </div>
      </div>

      {isLoading && (
        <div className="glass-card" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
          <Sparkles size={32} className="animate-spin" color="var(--primary)" style={{ margin: '0 auto 1rem' }} />
          <div>Extracting candidates, pairwise features, and model probabilities...</div>
        </div>
      )}

      {error && (
        <div className="glass-card" style={{ borderLeft: '4px solid var(--rose)', padding: '1.5rem', color: 'var(--rose)' }}>
          <div style={{ fontWeight: 700, marginBottom: '0.25rem' }}>Lookup Error</div>
          <div>{error}</div>
        </div>
      )}

      {entityData && !isLoading && (
        <>
          {/* Step 1: Source 1 Record Card */}
          <div className="glass-panel" style={{ padding: '1.5rem 2rem', borderLeft: '4px solid var(--primary)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.15rem', fontWeight: 800, color: 'var(--cyan)' }}>
                  {entityData.source1_entity_id}
                </div>
                <span className="badge badge-indigo">Source 1 Reference</span>
                {entityData.is_singleton ? (
                  <span className="badge badge-amber">Predicted Singleton (0 Matches)</span>
                ) : (
                  <span className="badge badge-emerald">
                    {entityData.matched_entity_ids.length} Final Match{entityData.matched_entity_ids.length > 1 ? 'es' : ''}
                  </span>
                )}
              </div>

              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                {entityData.candidate_entity_ids.length} Blocking Candidates Evaluated
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem' }}>
              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
                <Building2 size={20} color="var(--primary)" style={{ marginTop: '0.15rem', flexShrink: 0 }} />
                <div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                    Business Name
                  </div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.2rem' }}>
                    {entityData.business_name || '<Empty Name>'}
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
                <MapPin size={20} color="var(--emerald)" style={{ marginTop: '0.15rem', flexShrink: 0 }} />
                <div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                    Business Address
                  </div>
                  <div style={{ fontSize: '0.95rem', fontWeight: 500, color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                    {entityData.business_address || '<Omitted Address>'}
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
                <Globe size={20} color="var(--cyan)" style={{ marginTop: '0.15rem', flexShrink: 0 }} />
                <div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                    Jurisdiction / Country
                  </div>
                  <div style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.2rem' }}>
                    {entityData.country}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Step 2 -> 5: Candidate Records, Features, Probabilities, and Final Decision */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <h3 style={{ fontSize: '1.15rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <GitCompare size={18} color="var(--cyan)" />
                <span>Candidate Records Resolution & Decision Gating</span>
              </h3>
              <span className="badge badge-indigo">Decision Threshold: τ = 0.88</span>
            </div>

            {(!entityData.candidate_details || entityData.candidate_details.length === 0) ? (
              <div className="glass-card" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                No blocking candidates were retrieved for this entity. Correctly emitted as singleton.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {entityData.candidate_details.map((cand, idx) => {
                  const isAccepted = cand.is_match;
                  return (
                    <div
                      key={idx}
                      className="glass-card"
                      style={{
                        padding: '1.25rem 1.5rem',
                        borderLeft: isAccepted ? '4px solid var(--emerald)' : '4px solid rgba(255, 255, 255, 0.1)',
                        backgroundColor: isAccepted ? 'rgba(16, 185, 129, 0.05)' : 'var(--bg-card)',
                      }}
                    >
                      {/* Top Row: IDs, Sources, and Final Decision Badge */}
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.85rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.95rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                            {cand.entity_id}
                          </span>
                          <span className={cand.source === 'Source 2' ? 'badge badge-indigo' : 'badge badge-cyan'}>
                            {cand.source}
                          </span>
                          <span className="badge badge-indigo">{cand.country}</span>
                        </div>

                        {/* Final Decision Badge */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                            Model Score: <strong>{(cand.model_probability * 100).toFixed(1)}%</strong>
                          </span>
                          {isAccepted ? (
                            <span className="badge badge-emerald" style={{ padding: '0.35rem 0.75rem', fontSize: '0.8rem' }}>
                              <CheckCircle2 size={14} />
                              <span>MATCH ACCEPTED</span>
                            </span>
                          ) : (
                            <span className="badge badge-rose" style={{ padding: '0.35rem 0.75rem', fontSize: '0.8rem', opacity: 0.75 }}>
                              <XCircle size={14} />
                              <span>REJECTED (Score &lt; 0.88)</span>
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Candidate Record Content */}
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', marginBottom: '1rem' }}>
                        <div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Candidate Business Name</div>
                          <div style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)', marginTop: '0.15rem' }}>
                            {cand.business_name}
                          </div>
                        </div>
                        <div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Candidate Business Address</div>
                          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>
                            {cand.business_address || '<No Address Provided>'}
                          </div>
                        </div>
                      </div>

                      {/* Similarity Feature Breakdown */}
                      <div
                        style={{
                          backgroundColor: 'rgba(0, 0, 0, 0.25)',
                          borderRadius: 'var(--radius-sm)',
                          padding: '0.85rem 1rem',
                          display: 'grid',
                          gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
                          gap: '0.75rem',
                          border: '1px solid var(--border-subtle)',
                        }}
                      >
                        <div>
                          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Name Sort Sim</div>
                          <div style={{ fontSize: '0.95rem', fontWeight: 700, color: cand.name_similarity > 0.85 ? 'var(--emerald)' : 'var(--text-secondary)' }}>
                            {(cand.name_similarity * 100).toFixed(1)}%
                          </div>
                        </div>

                        <div>
                          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Address Sim</div>
                          <div style={{ fontSize: '0.95rem', fontWeight: 700, color: cand.address_similarity > 0.8 ? 'var(--emerald)' : 'var(--text-secondary)' }}>
                            {(cand.address_similarity * 100).toFixed(1)}%
                          </div>
                        </div>

                        <div>
                          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Numeric Overlap</div>
                          <div style={{ fontSize: '0.95rem', fontWeight: 700, color: cand.numeric_overlap > 0.5 ? 'var(--emerald)' : 'var(--text-secondary)' }}>
                            {(cand.numeric_overlap * 100).toFixed(1)}%
                          </div>
                        </div>

                        <div>
                          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Composite Score</div>
                          <div style={{ fontSize: '0.95rem', fontWeight: 700, color: cand.composite_score > 0.85 ? 'var(--cyan)' : 'var(--text-secondary)' }}>
                            {(cand.composite_score * 100).toFixed(1)}%
                          </div>
                        </div>

                        <div>
                          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Model Probability</div>
                          <div style={{ fontSize: '0.95rem', fontWeight: 800, color: isAccepted ? 'var(--emerald)' : 'var(--rose)' }}>
                            {(cand.model_probability * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

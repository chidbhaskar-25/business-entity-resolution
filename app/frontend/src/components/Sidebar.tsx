import React from 'react';
import {
  LayoutDashboard,
  Database,
  GitCompare,
  FlaskConical,
  ShieldCheck,
  Send,
  Cpu,
  Sliders,
  Layers,
} from 'lucide-react';

export type PageId =
  | 'dashboard'
  | 'dataset'
  | 'matcher'
  | 'evaluation'
  | 'experiments'
  | 'validation'
  | 'submission';

interface SidebarProps {
  activePage: PageId;
  onSelectPage: (page: PageId) => void;
  status: string;
  activeModel: string;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activePage,
  onSelectPage,
  status,
  activeModel,
}) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'dataset', label: 'Dataset Explorer', icon: Database },
    { id: 'matcher', label: 'Match Explorer', icon: GitCompare },
    { id: 'evaluation', label: 'F0.5 Evaluation', icon: Sliders },
    { id: 'experiments', label: 'Experiments', icon: FlaskConical },
    { id: 'validation', label: 'Validation', icon: ShieldCheck },
    { id: 'submission', label: 'Submission', icon: Send },
  ];

  return (
    <aside
      style={{
        width: '260px',
        backgroundColor: 'var(--bg-sidebar)',
        borderRight: '1px solid var(--border-subtle)',
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        position: 'sticky',
        top: 0,
        zIndex: 50,
      }}
    >
      {/* Brand Header */}
      <div
        style={{
          padding: '1.5rem 1.25rem',
          borderBottom: '1px solid var(--border-subtle)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
        }}
      >
        <div
          style={{
            width: '38px',
            height: '38px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #6366f1, #06b6d4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 15px rgba(99, 102, 241, 0.4)',
          }}
        >
          <Layers size={22} color="#ffffff" />
        </div>
        <div>
          <div style={{ fontSize: '1.05rem', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
            EntityResolver<span style={{ color: 'var(--cyan)' }}>.AI</span>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 600 }}>
            Amazon ML Challenge 2026
          </div>
        </div>
      </div>

      {/* Navigation Links */}
      <nav style={{ padding: '1rem 0.75rem', display: 'flex', flexDirection: 'column', gap: '0.35rem', flex: 1 }}>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activePage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectPage(item.id as PageId)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
                padding: '0.7rem 0.9rem',
                borderRadius: 'var(--radius-md)',
                background: isActive
                  ? 'linear-gradient(90deg, rgba(99, 102, 241, 0.2), rgba(99, 102, 241, 0.05))'
                  : 'transparent',
                color: isActive ? '#ffffff' : 'var(--text-secondary)',
                border: isActive ? '1px solid rgba(99, 102, 241, 0.35)' : '1px solid transparent',
                fontWeight: isActive ? 600 : 500,
                fontSize: '0.875rem',
                cursor: 'pointer',
                textAlign: 'left',
                width: '100%',
                transition: 'all 0.15s ease',
              }}
            >
              <Icon size={18} color={isActive ? 'var(--cyan)' : 'var(--text-muted)'} />
              <span>{item.label}</span>
              {isActive && (
                <div
                  style={{
                    marginLeft: 'auto',
                    width: '6px',
                    height: '6px',
                    borderRadius: '50%',
                    backgroundColor: 'var(--cyan)',
                    boxShadow: '0 0 8px var(--cyan)',
                  }}
                />
              )}
            </button>
          );
        })}
      </nav>

      {/* System Status Footer */}
      <div
        style={{
          padding: '1rem 1.25rem',
          borderTop: '1px solid var(--border-subtle)',
          backgroundColor: 'rgba(0,0,0,0.2)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
          <div
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: status === 'healthy' ? 'var(--emerald)' : 'var(--rose)',
              boxShadow: status === 'healthy' ? '0 0 8px var(--emerald)' : 'none',
            }}
          />
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
            Backend: {status === 'healthy' ? 'Online' : 'Offline'}
          </span>
        </div>
        <div
          style={{
            fontSize: '0.7rem',
            color: 'var(--text-muted)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.35rem',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
          title={activeModel}
        >
          <Cpu size={12} />
          <span>{activeModel.split(' ')[0] || 'HistGradBoost'}</span>
        </div>
      </div>
    </aside>
  );
};

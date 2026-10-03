'use client';

import React from 'react';
import { ShieldAlert, Cpu, Sparkles, Sliders, ExternalLink, Activity } from 'lucide-react';
import { BackboneType } from '../lib/types';
import { BACKBONE_REGISTRY } from '../lib/constants';

interface NavbarProps {
  selectedBackbone: BackboneType;
  onSelectBackbone: (backbone: BackboneType) => void;
  onOpenSettings: () => void;
  onOpenResearchInfo: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  selectedBackbone,
  onSelectBackbone,
  onOpenSettings,
  onOpenResearchInfo,
}) => {
  return (
    <header style={{
      borderBottom: '1px solid var(--border-subtle)',
      backgroundColor: 'rgba(7, 9, 14, 0.85)',
      backdropFilter: 'blur(20px)',
      position: 'sticky',
      top: 0,
      zIndex: 50,
      padding: '12px 24px',
    }}>
      <div style={{
        maxWidth: 1400,
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: 16,
      }}>
        {/* Logo & Branding */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 42,
            height: 42,
            borderRadius: 12,
            background: 'linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 20px rgba(6, 182, 212, 0.4)',
          }}>
            <ShieldAlert size={24} color="#ffffff" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <h1 className="font-display" style={{
                fontSize: '1.25rem',
                fontWeight: 700,
                letterSpacing: '-0.02em',
                margin: 0,
              }}>
                Deepfake<span style={{ color: '#06b6d4' }}>Forensic</span> AI
              </h1>
              <span className="badge badge-cyan" style={{ fontSize: '0.65rem' }}>
                <Activity size={11} /> Vercel Ready
              </span>
            </div>
            <p style={{
              fontSize: '0.75rem',
              color: 'var(--text-muted)',
              margin: 0,
            }}>
              Multi-Backbone Video Generalization & Diffusion Defense
            </p>
          </div>
        </div>

        {/* Center: Backbone Selector Tabs */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          backgroundColor: 'rgba(15, 20, 32, 0.8)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 12,
          padding: 4,
          gap: 4,
        }}>
          {(Object.keys(BACKBONE_REGISTRY) as BackboneType[]).map((key) => {
            const info = BACKBONE_REGISTRY[key];
            const isSelected = selectedBackbone === key;
            return (
              <button
                key={key}
                id={`btn-backbone-${key}`}
                onClick={() => onSelectBackbone(key)}
                style={{
                  background: isSelected ? 'linear-gradient(135deg, rgba(79, 70, 229, 0.8) 0%, rgba(6, 182, 212, 0.8) 100%)' : 'transparent',
                  color: isSelected ? '#ffffff' : 'var(--text-muted)',
                  border: isSelected ? '1px solid rgba(255, 255, 255, 0.2)' : '1px solid transparent',
                  borderRadius: 8,
                  padding: '6px 12px',
                  fontSize: '0.8rem',
                  fontFamily: 'var(--font-display)',
                  fontWeight: isSelected ? 600 : 500,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  transition: 'all 0.2s ease',
                  boxShadow: isSelected ? '0 0 14px rgba(6, 182, 212, 0.35)' : 'none',
                }}
              >
                {key === 'fusion' && <Sparkles size={13} color={isSelected ? '#ffffff' : '#06b6d4'} />}
                {key === 'resnet' && <Cpu size={13} />}
                {key === 'clip' && <ShieldAlert size={13} />}
                <span>{info.name.split(' ')[0]}</span>
                {key === 'fusion' && (
                  <span style={{
                    fontSize: '0.65rem',
                    backgroundColor: 'rgba(255, 255, 255, 0.2)',
                    padding: '1px 5px',
                    borderRadius: 4,
                  }}>
                    +8.7% F1
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {/* Right Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button
            id="btn-nav-research"
            className="btn-secondary"
            onClick={onOpenResearchInfo}
            title="View Research Paper Findings"
          >
            <Sparkles size={15} color="#38bdf8" />
            <span>Research Findings</span>
          </button>
          <button
            id="btn-nav-settings"
            className="btn-secondary"
            onClick={onOpenSettings}
            title="Configure Forensic Engine"
          >
            <Sliders size={15} />
            <span>Config</span>
          </button>
        </div>
      </div>
    </header>
  );
};

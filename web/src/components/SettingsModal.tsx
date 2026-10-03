'use client';

import React from 'react';
import { X, Sliders, CheckCircle, RotateCcw, Link2, Sparkles } from 'lucide-react';
import { AnalysisConfig, BackboneType } from '../lib/types';
import { BACKBONE_REGISTRY, DEFAULT_CONFIG } from '../lib/constants';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: AnalysisConfig;
  onChangeConfig: (newConfig: AnalysisConfig) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  config,
  onChangeConfig,
}) => {
  if (!isOpen) return null;

  const resetDefaults = () => {
    onChangeConfig(DEFAULT_CONFIG);
  };

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.85)',
      backdropFilter: 'blur(12px)',
      zIndex: 100,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px',
    }}>
      <div className="glass-panel" style={{
        maxWidth: 580,
        width: '100%',
        padding: '28px',
        backgroundColor: '#0b0f19',
        border: '1px solid rgba(255, 255, 255, 0.15)',
        position: 'relative',
      }}>
        {/* Close Button */}
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: 20,
            right: 20,
            background: 'none',
            border: 'none',
            color: 'var(--text-muted)',
            cursor: 'pointer',
          }}
        >
          <X size={20} />
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
          <Sliders size={20} color="#06b6d4" />
          <h2 className="font-display" style={{ fontSize: '1.25rem', fontWeight: 700, margin: 0 }}>
            Forensic Engine Configuration
          </h2>
        </div>

        {/* Section 1: Sampling Rate */}
        <div style={{ marginBottom: 20 }}>
          <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)', display: 'block', marginBottom: 6 }}>
            Temporal Sampling Rate (FPS)
          </label>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '0 0 10px 0' }}>
            Determines how many frames are sampled per second from the video. Matches the research paper standard of 1 fps.
          </p>
          <div style={{ display: 'flex', gap: 8 }}>
            {[0.5, 1, 2].map((fps) => (
              <button
                key={fps}
                className={config.samplingFps === fps ? 'btn-primary' : 'btn-secondary'}
                onClick={() => onChangeConfig({ ...config, samplingFps: fps })}
                style={{ fontSize: '0.8rem', padding: '6px 14px' }}
              >
                {fps} FPS {fps === 1 && '(Standard)'}
              </button>
            ))}
          </div>
        </div>

        {/* Section 2: Sensitivity Threshold */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
            <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)' }}>
              Deepfake Detection Sensitivity Threshold
            </label>
            <span className="font-mono" style={{ fontSize: '0.85rem', color: '#38bdf8', fontWeight: 600 }}>
              {(config.sensitivityThreshold * 100).toFixed(0)}%
            </span>
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '0 0 8px 0' }}>
            Probability cutoff above which a frame is flagged as a synthetic manipulation.
          </p>
          <input
            type="range"
            min="0.30"
            max="0.70"
            step="0.05"
            value={config.sensitivityThreshold}
            onChange={(e) => onChangeConfig({ ...config, sensitivityThreshold: parseFloat(e.target.value) })}
            style={{ width: '100%', accentColor: '#06b6d4' }}
          />
        </div>

        {/* Section 3: Optional Python Backend API Relay */}
        <div style={{
          backgroundColor: 'rgba(15, 23, 42, 0.6)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 10,
          padding: 16,
          marginBottom: 24,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Link2 size={16} color="#818cf8" />
              <label style={{ fontSize: '0.85rem', fontWeight: 600, color: '#f8fafc', margin: 0 }}>
                Python PyTorch Backend Relay
              </label>
            </div>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: '0.75rem' }}>
              <input
                type="checkbox"
                checked={config.usePythonApi}
                onChange={(e) => onChangeConfig({ ...config, usePythonApi: e.target.checked })}
                style={{ accentColor: '#4f46e5' }}
              />
              <span>Enable Relay</span>
            </label>
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '0 0 10px 0' }}>
            When running locally or on GPU server, you can forward frames directly to your PyTorch FastAPI server.
          </p>
          <input
            type="text"
            value={config.pythonApiUrl}
            disabled={!config.usePythonApi}
            onChange={(e) => onChangeConfig({ ...config, pythonApiUrl: e.target.value })}
            placeholder="http://localhost:8000/predict"
            style={{
              width: '100%',
              backgroundColor: 'rgba(7, 9, 14, 0.8)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 6,
              padding: '8px 12px',
              color: '#ffffff',
              fontSize: '0.8rem',
              fontFamily: 'var(--font-mono)',
            }}
          />
        </div>

        {/* Footer */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <button className="btn-secondary" onClick={resetDefaults} style={{ fontSize: '0.8rem' }}>
            <RotateCcw size={14} />
            <span>Reset Defaults</span>
          </button>
          <button className="btn-primary" onClick={onClose} style={{ fontSize: '0.85rem' }}>
            <CheckCircle size={15} />
            <span>Apply & Close</span>
          </button>
        </div>
      </div>
    </div>
  );
};

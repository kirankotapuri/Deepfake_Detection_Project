'use client';

import React from 'react';
import { Layers, Sparkles, Cpu, ShieldAlert, Award } from 'lucide-react';
import { BackboneType, VideoAnalysisResult } from '../lib/types';
import { BACKBONE_REGISTRY } from '../lib/constants';

interface BackboneContributionCardProps {
  result: VideoAnalysisResult;
  selectedBackbone: BackboneType;
  onSelectBackbone: (backbone: BackboneType) => void;
}

export const BackboneContributionCard: React.FC<BackboneContributionCardProps> = ({
  result,
  selectedBackbone,
  onSelectBackbone,
}) => {
  return (
    <div className="glass-panel" style={{ padding: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 34,
            height: 34,
            borderRadius: 8,
            backgroundColor: 'rgba(99, 102, 241, 0.15)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#818cf8',
          }}>
            <Layers size={18} />
          </div>
          <div>
            <h3 className="font-display" style={{ fontSize: '1.1rem', fontWeight: 600, margin: 0 }}>
              Multi-Backbone Consensus & Generalization Matrix
            </h3>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: 0 }}>
              Comparative predictions across ResNet50, CLIP, DINOv2, and Learned Weighted Fusion
            </p>
          </div>
        </div>

        <span className="badge badge-indigo" style={{ fontSize: '0.7rem' }}>
          <Award size={12} /> Paper Benchmark Comparison
        </span>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
        gap: 12,
      }}>
        {(Object.keys(BACKBONE_REGISTRY) as BackboneType[]).map((key) => {
          const info = BACKBONE_REGISTRY[key];
          const comparison = result.backboneComparison[key];
          const isSelected = selectedBackbone === key;
          const fakeProbPct = Math.round(comparison.fakeProb * 100);
          const isFake = comparison.verdict === 'FAKE';

          return (
            <div
              key={key}
              id={`card-backbone-${key}`}
              onClick={() => onSelectBackbone(key)}
              style={{
                backgroundColor: isSelected ? 'rgba(30, 41, 59, 0.7)' : 'rgba(15, 20, 32, 0.6)',
                border: `1px solid ${isSelected ? info.color : 'var(--border-subtle)'}`,
                borderRadius: 12,
                padding: '16px',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                position: 'relative',
                boxShadow: isSelected ? `0 0 20px ${info.color}33` : 'none',
              }}
            >
              {key === 'fusion' && (
                <div style={{
                  position: 'absolute',
                  top: 10,
                  right: 10,
                  backgroundColor: 'rgba(6, 182, 212, 0.15)',
                  border: '1px solid rgba(6, 182, 212, 0.4)',
                  color: '#38bdf8',
                  fontSize: '0.65rem',
                  fontWeight: 700,
                  padding: '2px 6px',
                  borderRadius: 4,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 3,
                }}>
                  <Sparkles size={10} /> Top F1 (+8.7%)
                </div>
              )}

              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                {key === 'fusion' && <Sparkles size={16} color={info.color} />}
                {key === 'resnet' && <Cpu size={16} color={info.color} />}
                {key === 'clip' && <ShieldAlert size={16} color={info.color} />}
                {key === 'dinov2' && <Layers size={16} color={info.color} />}
                <span className="font-display" style={{ fontSize: '0.95rem', fontWeight: 600 }}>
                  {info.name}
                </span>
              </div>

              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 12, minHeight: 32 }}>
                {info.tagline}
              </p>

              {/* Probability bar */}
              <div style={{ marginBottom: 10 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: 4 }}>
                  <span style={{ color: 'var(--text-dim)' }}>Synthetic Probability</span>
                  <span className="font-mono" style={{ fontWeight: 700, color: isFake ? '#fb7185' : '#34d399' }}>
                    {fakeProbPct}% ({comparison.verdict})
                  </span>
                </div>
                <div style={{ height: 6, backgroundColor: 'rgba(255, 255, 255, 0.08)', borderRadius: 3, overflow: 'hidden' }}>
                  <div style={{
                    width: `${fakeProbPct}%`,
                    height: '100%',
                    backgroundColor: isFake ? '#f43f5e' : '#10b981',
                    borderRadius: 3,
                  }} />
                </div>
              </div>

              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                paddingTop: 8,
                borderTop: '1px solid rgba(255, 255, 255, 0.05)',
                fontSize: '0.7rem',
                color: 'var(--text-dim)',
              }}>
                <span>Dim: {info.dimensions}d</span>
                <span>Benchmark Acc: <strong style={{ color: '#f8fafc' }}>{info.testAccuracy}</strong></span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

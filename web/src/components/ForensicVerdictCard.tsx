'use client';

import React from 'react';
import { ShieldAlert, ShieldCheck, AlertTriangle, FileText, Share2, Layers } from 'lucide-react';
import { VideoAnalysisResult } from '../lib/types';
import { BACKBONE_REGISTRY } from '../lib/constants';

interface ForensicVerdictCardProps {
  result: VideoAnalysisResult;
  onOpenReport: () => void;
}

export const ForensicVerdictCard: React.FC<ForensicVerdictCardProps> = ({
  result,
  onOpenReport,
}) => {
  const isFake = result.overallVerdict === 'DEEPFAKE';
  const isReal = result.overallVerdict === 'AUTHENTIC';
  const isSuspicious = result.overallVerdict === 'SUSPICIOUS';

  // SVG Gauge calculations
  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (result.overallFakeProbability * circumference);

  const verdictColor = isFake ? 'var(--fake-red)' : isReal ? 'var(--real-green)' : 'var(--warning-amber)';
  const verdictGlow = isFake ? 'var(--fake-red-glow)' : isReal ? 'var(--real-green-glow)' : 'var(--warning-glow)';

  return (
    <div
      className="glass-panel"
      style={{
        padding: '28px',
        border: `1px solid ${verdictColor}`,
        boxShadow: `0 0 35px ${verdictGlow}`,
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: 24,
      }}>
        {/* Left: Animated Circular Gauge */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 20,
        }}>
          <div style={{ position: 'relative', width: 160, height: 160 }}>
            <svg width="160" height="160" style={{ transform: 'rotate(-90deg)' }}>
              {/* Background circle */}
              <circle
                cx="80"
                cy="80"
                r={radius}
                stroke="rgba(255, 255, 255, 0.08)"
                strokeWidth="12"
                fill="none"
              />
              {/* Progress circle */}
              <circle
                cx="80"
                cy="80"
                r={radius}
                stroke={isFake ? 'url(#fakeGradient)' : 'url(#realGradient)'}
                strokeWidth="12"
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="round"
                fill="none"
                style={{ transition: 'stroke-dashoffset 1s ease' }}
              />
              <defs>
                <linearGradient id="fakeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#f43f5e" />
                  <stop offset="100%" stopColor="#f97316" />
                </linearGradient>
                <linearGradient id="realGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#10b981" />
                  <stop offset="100%" stopColor="#06b6d4" />
                </linearGradient>
              </defs>
            </svg>

            {/* Inner Center Text */}
            <div style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <span className="font-mono" style={{
                fontSize: '1.8rem',
                fontWeight: 800,
                color: verdictColor,
                lineHeight: 1,
              }}>
                {(result.overallFakeProbability * 100).toFixed(1)}%
              </span>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 4, textTransform: 'uppercase' }}>
                Synthetic Prob
              </span>
            </div>
          </div>

          {/* Verdict Details */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
              {isFake && (
                <div style={{
                  padding: '4px 10px',
                  borderRadius: 6,
                  backgroundColor: 'rgba(244, 63, 94, 0.2)',
                  border: '1px solid var(--fake-red)',
                  color: '#fb7185',
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                }}>
                  <ShieldAlert size={14} />
                  <span>DEEPFAKE DETECTED</span>
                </div>
              )}
              {isReal && (
                <div style={{
                  padding: '4px 10px',
                  borderRadius: 6,
                  backgroundColor: 'rgba(16, 185, 129, 0.2)',
                  border: '1px solid var(--real-green)',
                  color: '#34d399',
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                }}>
                  <ShieldCheck size={14} />
                  <span>AUTHENTIC MEDIA</span>
                </div>
              )}
              {isSuspicious && (
                <div style={{
                  padding: '4px 10px',
                  borderRadius: 6,
                  backgroundColor: 'rgba(245, 158, 11, 0.2)',
                  border: '1px solid var(--warning-amber)',
                  color: '#fbbf24',
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                }}>
                  <AlertTriangle size={14} />
                  <span>SUSPICIOUS ANOMALY</span>
                </div>
              )}
              <span className="badge badge-indigo">
                {result.confidenceScore}% Confidence
              </span>
            </div>

            <h2 className="font-display" style={{
              fontSize: '1.75rem',
              fontWeight: 800,
              letterSpacing: '-0.02em',
              marginBottom: 6,
            }}>
              {isFake ? 'Synthesized / Manipulated Video' : isReal ? 'Verified Authentic Camera Capture' : 'Anomalous Facial Patterns Detected'}
            </h2>

            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', margin: 0, maxWidth: 480 }}>
              Analyzed {result.totalFramesAnalyzed} video frames using{' '}
              <strong style={{ color: '#38bdf8' }}>{BACKBONE_REGISTRY[result.selectedBackbone].name}</strong>.
              Estimated generator fingerprint matches{' '}
              <span style={{ color: isFake ? '#fb7185' : '#34d399', fontWeight: 600 }}>
                {result.suspectedGenerator}
              </span>.
            </p>
          </div>
        </div>

        {/* Right: Key Forensic Biomarkers & Actions */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'flex-end',
          gap: 12,
        }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(2, 1fr)',
            gap: 8,
            backgroundColor: 'rgba(7, 9, 14, 0.6)',
            padding: 12,
            borderRadius: 10,
            border: '1px solid var(--border-subtle)',
          }}>
            <div>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', display: 'block' }}>Boundary Seam</span>
              <span className="font-mono" style={{ fontSize: '0.9rem', fontWeight: 600, color: result.summaryMetrics.boundarySeamScore > 0.4 ? '#fb7185' : '#34d399' }}>
                {(result.summaryMetrics.boundarySeamScore * 100).toFixed(1)}%
              </span>
            </div>
            <div>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', display: 'block' }}>Fourier 2D Peak</span>
              <span className="font-mono" style={{ fontSize: '0.9rem', fontWeight: 600, color: result.summaryMetrics.fftFrequencyAnomaly > 0.4 ? '#fb7185' : '#34d399' }}>
                {(result.summaryMetrics.fftFrequencyAnomaly * 100).toFixed(1)}%
              </span>
            </div>
            <div>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', display: 'block' }}>Chromatic Shift</span>
              <span className="font-mono" style={{ fontSize: '0.9rem', fontWeight: 600, color: result.summaryMetrics.chromaticInconsistency > 0.4 ? '#fb7185' : '#34d399' }}>
                {(result.summaryMetrics.chromaticInconsistency * 100).toFixed(1)}%
              </span>
            </div>
            <div>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', display: 'block' }}>Temporal Jitter</span>
              <span className="font-mono" style={{ fontSize: '0.9rem', fontWeight: 600, color: result.summaryMetrics.temporalJitter > 0.35 ? '#fb7185' : '#34d399' }}>
                {(result.summaryMetrics.temporalJitter * 100).toFixed(1)}%
              </span>
            </div>
          </div>

          <div style={{ display: 'flex', gap: 8 }}>
            <button
              id="btn-verdict-report"
              className="btn-primary"
              onClick={onOpenReport}
              style={{ fontSize: '0.85rem', padding: '8px 16px' }}
            >
              <FileText size={15} />
              <span>Export Full Forensic Report</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

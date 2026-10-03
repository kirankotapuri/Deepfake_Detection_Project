'use client';

import React from 'react';
import { Eye, Activity, Radio, Cpu, Sparkles } from 'lucide-react';
import { FrameAnalysis } from '../lib/types';

interface GradCAMViewerProps {
  frame: FrameAnalysis | null;
}

export const GradCAMViewer: React.FC<GradCAMViewerProps> = ({ frame }) => {
  if (!frame) return null;

  return (
    <div className="glass-panel" style={{ padding: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 34,
            height: 34,
            borderRadius: 8,
            backgroundColor: 'rgba(244, 63, 94, 0.15)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#fb7185',
          }}>
            <Eye size={18} />
          </div>
          <div>
            <h3 className="font-display" style={{ fontSize: '1.1rem', fontWeight: 600, margin: 0 }}>
              Explainability: GradCAM & Frequency Domain Decomposition
            </h3>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: 0 }}>
              Frame #{frame.frameIndex + 1} ({frame.timestamp.toFixed(2)}s) — Spatial Attention & Fourier Spectrum
            </p>
          </div>
        </div>

        <span className={`badge ${frame.isFake ? 'badge-fake' : 'badge-real'}`}>
          {frame.isFake ? 'ANOMALOUS ACTIVATION' : 'NATURAL FACIAL ATTENTION'}
        </span>
      </div>

      {/* Visual Modalities Triple Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
        gap: 16,
        marginBottom: 20,
      }}>
        {/* Modality 1: Preprocessed Face Crop */}
        <div style={{
          backgroundColor: 'rgba(7, 9, 14, 0.6)',
          borderRadius: 12,
          padding: 12,
          border: '1px solid var(--border-subtle)',
          textAlign: 'center',
        }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: 8 }}>
            1. Standardized 224×224 Face Crop
          </span>
          <div style={{
            width: '100%',
            aspectRatio: '1/1',
            borderRadius: 8,
            overflow: 'hidden',
            backgroundColor: '#000000',
            position: 'relative',
          }}>
            {frame.faceCropUrl ? (
              <img
                src={frame.faceCropUrl}
                alt="Face crop"
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />
            ) : (
              <div style={{ color: 'var(--text-dim)', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                No face detected
              </div>
            )}
          </div>
          <span style={{ fontSize: '0.68rem', color: 'var(--text-dim)', marginTop: 6, display: 'block' }}>
            Input to Frozen Backbones
          </span>
        </div>

        {/* Modality 2: GradCAM Attention Heatmap */}
        <div style={{
          backgroundColor: 'rgba(7, 9, 14, 0.6)',
          borderRadius: 12,
          padding: 12,
          border: '1px solid var(--border-subtle)',
          textAlign: 'center',
        }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#fb7185', display: 'block', marginBottom: 8 }}>
            2. ResNet50 GradCAM Heatmap
          </span>
          <div style={{
            width: '100%',
            aspectRatio: '1/1',
            borderRadius: 8,
            overflow: 'hidden',
            backgroundColor: '#000000',
            position: 'relative',
          }}>
            {frame.gradcamHeatmapUrl ? (
              <img
                src={frame.gradcamHeatmapUrl}
                alt="GradCAM heatmap"
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />
            ) : null}
          </div>
          <span style={{ fontSize: '0.68rem', color: 'var(--text-dim)', marginTop: 6, display: 'block' }}>
            Red/Hot = Critical Artifact Triggers
          </span>
        </div>

        {/* Modality 3: 2D Fourier Magnitude Spectrum */}
        <div style={{
          backgroundColor: 'rgba(7, 9, 14, 0.6)',
          borderRadius: 12,
          padding: 12,
          border: '1px solid var(--border-subtle)',
          textAlign: 'center',
        }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#38bdf8', display: 'block', marginBottom: 8 }}>
            3. 2D Fourier Magnitude Spectrum
          </span>
          <div style={{
            width: '100%',
            aspectRatio: '1/1',
            borderRadius: 8,
            overflow: 'hidden',
            backgroundColor: '#000000',
            position: 'relative',
          }}>
            {frame.fftSpectrumUrl ? (
              <img
                src={frame.fftSpectrumUrl}
                alt="Fourier spectrum"
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />
            ) : null}
          </div>
          <span style={{ fontSize: '0.68rem', color: 'var(--text-dim)', marginTop: 6, display: 'block' }}>
            Periodic Spikes = GAN Upsampling Footprint
          </span>
        </div>
      </div>

      {/* Frame Metrics Breakdown */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
        gap: 12,
        backgroundColor: 'rgba(15, 23, 42, 0.5)',
        padding: 14,
        borderRadius: 10,
        border: '1px solid var(--border-subtle)',
      }}>
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: 4 }}>
            <span style={{ color: 'var(--text-muted)' }}>Boundary Seam</span>
            <span className="font-mono" style={{ fontWeight: 600, color: frame.metrics.boundarySeamScore > 0.4 ? '#fb7185' : '#34d399' }}>
              {(frame.metrics.boundarySeamScore * 100).toFixed(1)}%
            </span>
          </div>
          <div style={{ height: 4, backgroundColor: 'rgba(255, 255, 255, 0.08)', borderRadius: 2 }}>
            <div style={{ width: `${frame.metrics.boundarySeamScore * 100}%`, height: '100%', backgroundColor: frame.metrics.boundarySeamScore > 0.4 ? '#f43f5e' : '#10b981' }} />
          </div>
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: 4 }}>
            <span style={{ color: 'var(--text-muted)' }}>FFT High-Freq Anomaly</span>
            <span className="font-mono" style={{ fontWeight: 600, color: frame.metrics.fftFrequencyAnomaly > 0.4 ? '#fb7185' : '#34d399' }}>
              {(frame.metrics.fftFrequencyAnomaly * 100).toFixed(1)}%
            </span>
          </div>
          <div style={{ height: 4, backgroundColor: 'rgba(255, 255, 255, 0.08)', borderRadius: 2 }}>
            <div style={{ width: `${frame.metrics.fftFrequencyAnomaly * 100}%`, height: '100%', backgroundColor: frame.metrics.fftFrequencyAnomaly > 0.4 ? '#f43f5e' : '#10b981' }} />
          </div>
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: 4 }}>
            <span style={{ color: 'var(--text-muted)' }}>Chromatic Drift</span>
            <span className="font-mono" style={{ fontWeight: 600, color: frame.metrics.chromaticInconsistency > 0.4 ? '#fb7185' : '#34d399' }}>
              {(frame.metrics.chromaticInconsistency * 100).toFixed(1)}%
            </span>
          </div>
          <div style={{ height: 4, backgroundColor: 'rgba(255, 255, 255, 0.08)', borderRadius: 2 }}>
            <div style={{ width: `${frame.metrics.chromaticInconsistency * 100}%`, height: '100%', backgroundColor: frame.metrics.chromaticInconsistency > 0.4 ? '#f43f5e' : '#10b981' }} />
          </div>
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: 4 }}>
            <span style={{ color: 'var(--text-muted)' }}>Temporal Jitter</span>
            <span className="font-mono" style={{ fontWeight: 600, color: frame.metrics.temporalJitter > 0.35 ? '#fb7185' : '#34d399' }}>
              {(frame.metrics.temporalJitter * 100).toFixed(1)}%
            </span>
          </div>
          <div style={{ height: 4, backgroundColor: 'rgba(255, 255, 255, 0.08)', borderRadius: 2 }}>
            <div style={{ width: `${frame.metrics.temporalJitter * 100}%`, height: '100%', backgroundColor: frame.metrics.temporalJitter > 0.35 ? '#f43f5e' : '#10b981' }} />
          </div>
        </div>
      </div>
    </div>
  );
};

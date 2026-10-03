'use client';

import React from 'react';
import { X, Download, Printer, ShieldAlert, ShieldCheck, CheckCircle2, FileText, Cpu, Layers } from 'lucide-react';
import { VideoAnalysisResult } from '../lib/types';
import { BACKBONE_REGISTRY } from '../lib/constants';

interface ReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  result: VideoAnalysisResult | null;
}

export const ReportModal: React.FC<ReportModalProps> = ({ isOpen, onClose, result }) => {
  if (!isOpen || !result) return null;

  const downloadJson = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(result, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `forensic_report_${result.fileName}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const isFake = result.overallVerdict === 'DEEPFAKE';

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
        maxWidth: 780,
        width: '100%',
        maxHeight: '90vh',
        overflowY: 'auto',
        padding: '32px',
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

        {/* Report Header */}
        <div style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: 20, marginBottom: 24 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
            <FileText size={22} color="#06b6d4" />
            <h2 className="font-display" style={{ fontSize: '1.4rem', fontWeight: 700, margin: 0 }}>
              Forensic Media Authentication Certificate
            </h2>
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0 }}>
            Automated Cross-Dataset Deepfake Generalization Verification System • SRM AP Research Platform
          </p>
        </div>

        {/* Certificate Metadata */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: 12,
          backgroundColor: 'rgba(15, 23, 42, 0.5)',
          padding: 16,
          borderRadius: 10,
          border: '1px solid var(--border-subtle)',
          marginBottom: 24,
          fontSize: '0.8rem',
        }}>
          <div>
            <span style={{ color: 'var(--text-dim)', display: 'block' }}>Target Video File</span>
            <strong style={{ color: '#f8fafc' }}>{result.fileName}</strong>
          </div>
          <div>
            <span style={{ color: 'var(--text-dim)', display: 'block' }}>Duration / Frames</span>
            <strong style={{ color: '#f8fafc' }}>{result.duration}s ({result.totalFramesAnalyzed} frames sampled)</strong>
          </div>
          <div>
            <span style={{ color: 'var(--text-dim)', display: 'block' }}>Active Backbone</span>
            <strong style={{ color: '#38bdf8' }}>{BACKBONE_REGISTRY[result.selectedBackbone].name}</strong>
          </div>
          <div>
            <span style={{ color: 'var(--text-dim)', display: 'block' }}>Analysis Timestamp</span>
            <strong style={{ color: '#f8fafc' }}>{new Date(result.analyzedAt).toLocaleString()}</strong>
          </div>
        </div>

        {/* Primary Verdict Banner */}
        <div style={{
          padding: '20px',
          borderRadius: 12,
          backgroundColor: isFake ? 'rgba(244, 63, 94, 0.12)' : 'rgba(16, 185, 129, 0.12)',
          border: `1px solid ${isFake ? '#f43f5e' : '#10b981'}`,
          display: 'flex',
          alignItems: 'center',
          gap: 16,
          marginBottom: 24,
        }}>
          {isFake ? <ShieldAlert size={36} color="#f43f5e" /> : <ShieldCheck size={36} color="#10b981" />}
          <div>
            <div style={{ fontSize: '0.75rem', fontWeight: 700, color: isFake ? '#fb7185' : '#34d399', textTransform: 'uppercase' }}>
              Final Determination
            </div>
            <h3 className="font-display" style={{ fontSize: '1.5rem', fontWeight: 800, margin: '2px 0 4px 0' }}>
              {isFake ? 'SYNTHETIC MEDIA DETECTED (DEEPFAKE)' : 'GENUINE AUTHENTIC MEDIA'}
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', margin: 0 }}>
              Calculated Synthetic Confidence: <strong>{(result.overallFakeProbability * 100).toFixed(1)}%</strong> ({result.confidenceScore}% certainty).
              Attributed source: <strong>{result.suspectedGenerator}</strong>.
            </p>
          </div>
        </div>

        {/* Multi-Backbone Consensus Table */}
        <h4 className="font-display" style={{ fontSize: '1rem', fontWeight: 600, marginBottom: 12 }}>
          Multi-Backbone Model Consensus
        </h4>
        <div style={{
          border: '1px solid var(--border-subtle)',
          borderRadius: 10,
          overflow: 'hidden',
          marginBottom: 24,
        }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
            <thead>
              <tr style={{ backgroundColor: 'rgba(15, 23, 42, 0.8)', borderBottom: '1px solid var(--border-subtle)' }}>
                <th style={{ padding: '10px 14px', textAlign: 'left' }}>Backbone Architecture</th>
                <th style={{ padding: '10px 14px', textAlign: 'left' }}>Dimensions</th>
                <th style={{ padding: '10px 14px', textAlign: 'left' }}>Synthetic Probability</th>
                <th style={{ padding: '10px 14px', textAlign: 'left' }}>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {(['fusion', 'resnet', 'clip', 'dinov2'] as const).map((key) => {
                const info = BACKBONE_REGISTRY[key];
                const cmp = result.backboneComparison[key];
                return (
                  <tr key={key} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                    <td style={{ padding: '10px 14px' }}>
                      <strong style={{ color: info.color }}>{info.name}</strong>
                    </td>
                    <td style={{ padding: '10px 14px', color: 'var(--text-muted)' }}>{info.dimensions}d</td>
                    <td style={{ padding: '10px 14px', fontFamily: 'var(--font-mono)' }}>
                      {(cmp.fakeProb * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: '10px 14px' }}>
                      <span className={`badge ${cmp.verdict === 'FAKE' ? 'badge-fake' : 'badge-real'}`} style={{ fontSize: '0.65rem' }}>
                        {cmp.verdict}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Biomarkers */}
        <h4 className="font-display" style={{ fontSize: '1rem', fontWeight: 600, marginBottom: 12 }}>
          Biometric Forensic Biomarkers
        </h4>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: 12,
          marginBottom: 28,
        }}>
          <div style={{ padding: 12, borderRadius: 8, backgroundColor: 'rgba(15, 23, 42, 0.6)', border: '1px solid var(--border-subtle)' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Boundary Seam Ratio</span>
            <div className="font-mono" style={{ fontSize: '1.1rem', fontWeight: 700, color: result.summaryMetrics.boundarySeamScore > 0.4 ? '#fb7185' : '#34d399', margin: '4px 0' }}>
              {(result.summaryMetrics.boundarySeamScore * 100).toFixed(1)}%
            </div>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
              Laplacian edge variance across facial perimeter stitching
            </span>
          </div>

          <div style={{ padding: 12, borderRadius: 8, backgroundColor: 'rgba(15, 23, 42, 0.6)', border: '1px solid var(--border-subtle)' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>2D Fourier Spectrum Peak</span>
            <div className="font-mono" style={{ fontSize: '1.1rem', fontWeight: 700, color: result.summaryMetrics.fftFrequencyAnomaly > 0.4 ? '#fb7185' : '#34d399', margin: '4px 0' }}>
              {(result.summaryMetrics.fftFrequencyAnomaly * 100).toFixed(1)}%
            </div>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
              High-frequency periodic energy spikes from GAN upsampling
            </span>
          </div>

          <div style={{ padding: 12, borderRadius: 8, backgroundColor: 'rgba(15, 23, 42, 0.6)', border: '1px solid var(--border-subtle)' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Temporal Continuity Index</span>
            <div className="font-mono" style={{ fontSize: '1.1rem', fontWeight: 700, color: result.summaryMetrics.temporalJitter > 0.35 ? '#fb7185' : '#34d399', margin: '4px 0' }}>
              {(result.summaryMetrics.temporalJitter * 100).toFixed(1)}%
            </div>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
              Cross-frame pixel fluctuation and landmark jitter
            </span>
          </div>
        </div>

        {/* Modal Actions */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
          <button className="btn-secondary" onClick={() => window.print()}>
            <Printer size={15} />
            <span>Print Report</span>
          </button>
          <button className="btn-primary" onClick={downloadJson}>
            <Download size={15} />
            <span>Download JSON Analysis</span>
          </button>
        </div>
      </div>
    </div>
  );
};

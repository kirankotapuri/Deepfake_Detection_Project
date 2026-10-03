'use client';

import React from 'react';
import { Clock, Film, AlertCircle } from 'lucide-react';
import { FrameAnalysis } from '../lib/types';

interface FrameTimelineScrubberProps {
  frames: FrameAnalysis[];
  activeFrame: FrameAnalysis | null;
  onSelectFrame: (frame: FrameAnalysis) => void;
}

export const FrameTimelineScrubber: React.FC<FrameTimelineScrubberProps> = ({
  frames,
  activeFrame,
  onSelectFrame,
}) => {
  if (frames.length === 0) return null;

  return (
    <div className="glass-panel" style={{ padding: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Film size={16} color="#38bdf8" />
          <h4 className="font-display" style={{ fontSize: '0.95rem', fontWeight: 600, margin: 0 }}>
            Temporal Frame Forensic Timeline ({frames.length} sampled frames)
          </h4>
        </div>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Click any frame thumbnail to inspect localized biometric activations
        </span>
      </div>

      {/* Probability Bar Chart Timeline */}
      <div style={{
        height: 48,
        display: 'flex',
        alignItems: 'flex-end',
        gap: 3,
        marginBottom: 8,
        backgroundColor: 'rgba(7, 9, 14, 0.5)',
        padding: '6px 8px',
        borderRadius: 8,
        border: '1px solid var(--border-subtle)',
      }}>
        {frames.map((f) => {
          const isSelected = activeFrame?.frameIndex === f.frameIndex;
          const heightPercent = Math.max(12, Math.round(f.fakeProbability * 100));
          return (
            <div
              key={`bar-${f.frameIndex}`}
              id={`timeline-bar-${f.frameIndex}`}
              onClick={() => onSelectFrame(f)}
              title={`Frame #${f.frameIndex + 1} (${f.timestamp.toFixed(1)}s): ${(f.fakeProbability * 100).toFixed(1)}% Fake`}
              style={{
                flex: 1,
                height: `${heightPercent}%`,
                backgroundColor: f.isFake ? '#f43f5e' : '#10b981',
                borderRadius: '3px 3px 0 0',
                cursor: 'pointer',
                opacity: isSelected ? 1.0 : 0.6,
                transform: isSelected ? 'scaleY(1.15)' : 'scaleY(1)',
                boxShadow: isSelected ? `0 0 10px ${f.isFake ? '#f43f5e' : '#10b981'}` : 'none',
                transition: 'all 0.15s ease',
              }}
            />
          );
        })}
      </div>

      {/* Frame Filmstrip Thumbnails */}
      <div style={{
        display: 'flex',
        gap: 8,
        overflowX: 'auto',
        paddingBottom: 6,
      }}>
        {frames.map((f) => {
          const isSelected = activeFrame?.frameIndex === f.frameIndex;
          return (
            <div
              key={`thumb-${f.frameIndex}`}
              id={`timeline-thumb-${f.frameIndex}`}
              onClick={() => onSelectFrame(f)}
              style={{
                flex: '0 0 76px',
                height: 76,
                borderRadius: 8,
                overflow: 'hidden',
                position: 'relative',
                cursor: 'pointer',
                border: isSelected
                  ? '2px solid #06b6d4'
                  : `1px solid ${f.isFake ? 'rgba(244, 63, 94, 0.4)' : 'rgba(16, 185, 129, 0.4)'}`,
                boxShadow: isSelected ? '0 0 16px rgba(6, 182, 212, 0.5)' : 'none',
                transition: 'all 0.2s ease',
              }}
            >
              <img
                src={f.faceCropUrl || f.dataUrl}
                alt={`Frame ${f.frameIndex}`}
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />

              {/* Timecode badge */}
              <div style={{
                position: 'absolute',
                bottom: 0,
                left: 0,
                right: 0,
                backgroundColor: 'rgba(0, 0, 0, 0.75)',
                fontSize: '0.6rem',
                fontFamily: 'var(--font-mono)',
                textAlign: 'center',
                color: '#ffffff',
                padding: '2px 0',
              }}>
                {f.timestamp.toFixed(1)}s
              </div>

              {/* Fake/Real pill */}
              <div style={{
                position: 'absolute',
                top: 3,
                right: 3,
                width: 8,
                height: 8,
                borderRadius: '50%',
                backgroundColor: f.isFake ? '#f43f5e' : '#10b981',
                boxShadow: `0 0 6px ${f.isFake ? '#f43f5e' : '#10b981'}`,
              }} />
            </div>
          );
        })}
      </div>
    </div>
  );
};

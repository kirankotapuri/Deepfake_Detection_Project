'use client';

import React, { useRef, useState, useEffect } from 'react';
import { Play, Pause, Eye, EyeOff, Sliders, Maximize2, ShieldAlert } from 'lucide-react';
import { FrameAnalysis } from '../lib/types';

interface VideoPlayerWithInspectorProps {
  videoBlobUrl: string | null;
  activeFrame: FrameAnalysis | null;
  allFrames: FrameAnalysis[];
  onSeekFrame: (frame: FrameAnalysis) => void;
}

export const VideoPlayerWithInspector: React.FC<VideoPlayerWithInspectorProps> = ({
  videoBlobUrl,
  activeFrame,
  allFrames,
  onSeekFrame,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showBoundingBox, setShowBoundingBox] = useState(true);
  const [showGradCAM, setShowGradCAM] = useState(true);
  const [gradcamAlpha, setGradcamAlpha] = useState(0.55);
  const [currentTime, setCurrentTime] = useState(0);

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
    } else {
      videoRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleTimeUpdate = () => {
    if (!videoRef.current) return;
    const t = videoRef.current.currentTime;
    setCurrentTime(t);

    // Sync active frame with closest frame in allFrames
    if (allFrames.length > 0) {
      let closest = allFrames[0];
      let minDiff = Math.abs(allFrames[0].timestamp - t);
      for (const f of allFrames) {
        const diff = Math.abs(f.timestamp - t);
        if (diff < minDiff) {
          minDiff = diff;
          closest = f;
        }
      }
      if (closest.frameIndex !== activeFrame?.frameIndex) {
        onSeekFrame(closest);
      }
    }
  };

  useEffect(() => {
    if (activeFrame && videoRef.current && Math.abs(videoRef.current.currentTime - activeFrame.timestamp) > 0.3) {
      videoRef.current.currentTime = activeFrame.timestamp;
    }
  }, [activeFrame]);

  return (
    <div className="glass-panel" style={{ padding: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14, flexWrap: 'wrap', gap: 10 }}>
        <div>
          <h3 className="font-display" style={{ fontSize: '1.1rem', fontWeight: 600, margin: 0 }}>
            Live Forensics Video Inspector
          </h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: 0 }}>
            Synchronized playback with facial bounding box and GradCAM attention heatmap overlay
          </p>
        </div>

        {/* Overlay Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button
            id="btn-toggle-bbox"
            className="btn-secondary"
            onClick={() => setShowBoundingBox(!showBoundingBox)}
            style={{
              fontSize: '0.8rem',
              padding: '6px 12px',
              color: showBoundingBox ? '#38bdf8' : 'var(--text-dim)',
              borderColor: showBoundingBox ? 'rgba(56, 189, 248, 0.4)' : 'var(--border-subtle)',
            }}
          >
            {showBoundingBox ? <Eye size={14} /> : <EyeOff size={14} />}
            <span>Face Box</span>
          </button>

          <button
            id="btn-toggle-gradcam"
            className="btn-secondary"
            onClick={() => setShowGradCAM(!showGradCAM)}
            style={{
              fontSize: '0.8rem',
              padding: '6px 12px',
              color: showGradCAM ? '#fb7185' : 'var(--text-dim)',
              borderColor: showGradCAM ? 'rgba(244, 63, 94, 0.4)' : 'var(--border-subtle)',
            }}
          >
            <ShieldAlert size={14} />
            <span>GradCAM Overlay</span>
          </button>

          {showGradCAM && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              <span>Heatmap Alpha:</span>
              <input
                id="slider-gradcam-alpha"
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={gradcamAlpha}
                onChange={(e) => setGradcamAlpha(parseFloat(e.target.value))}
                style={{ width: 70, accentColor: '#f43f5e' }}
              />
            </div>
          )}
        </div>
      </div>

      {/* Main Video & Inspection Canvas Container */}
      <div style={{
        position: 'relative',
        width: '100%',
        backgroundColor: '#05070c',
        borderRadius: 12,
        overflow: 'hidden',
        border: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: 380,
      }}>
        {videoBlobUrl ? (
          <video
            ref={videoRef}
            src={videoBlobUrl}
            onTimeUpdate={handleTimeUpdate}
            onEnded={() => setIsPlaying(false)}
            playsInline
            style={{
              width: '100%',
              maxHeight: 520,
              objectFit: 'contain',
              display: 'block',
            }}
          />
        ) : (
          <div style={{ color: 'var(--text-dim)', textAlign: 'center', padding: 40 }}>
            No video loaded. Upload a video file or pick a benchmark sample above.
          </div>
        )}

        {/* Dynamic Canvas Overlays for Active Frame */}
        {activeFrame && activeFrame.faceBox && showBoundingBox && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              pointerEvents: 'none',
            }}
          >
            {/* The Bounding Box */}
            <div
              style={{
                position: 'absolute',
                left: `${(activeFrame.faceBox.x / (activeFrame.dataUrl ? 640 : 100)) * 100}%`,
                top: `${(activeFrame.faceBox.y / (activeFrame.dataUrl ? 480 : 100)) * 100}%`,
                width: `${(activeFrame.faceBox.width / (activeFrame.dataUrl ? 640 : 100)) * 100}%`,
                height: `${(activeFrame.faceBox.height / (activeFrame.dataUrl ? 480 : 100)) * 100}%`,
                border: `2px solid ${activeFrame.isFake ? '#f43f5e' : '#10b981'}`,
                boxShadow: `0 0 15px ${activeFrame.isFake ? 'rgba(244, 63, 94, 0.4)' : 'rgba(16, 185, 129, 0.4)'}`,
                borderRadius: 8,
              }}
            >
              {/* Corner Tag */}
              <div style={{
                position: 'absolute',
                top: -24,
                left: -2,
                backgroundColor: activeFrame.isFake ? '#f43f5e' : '#10b981',
                color: '#ffffff',
                fontSize: '0.65rem',
                fontFamily: 'var(--font-mono)',
                fontWeight: 700,
                padding: '2px 6px',
                borderRadius: '4px 4px 0 0',
                display: 'flex',
                alignItems: 'center',
                gap: 4,
              }}>
                <span>{activeFrame.isFake ? 'FAKE' : 'REAL'}</span>
                <span>{(activeFrame.fakeProbability * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>
        )}

        {/* PIP Face Crop Inspection in corner */}
        {activeFrame && activeFrame.faceCropUrl && (
          <div style={{
            position: 'absolute',
            bottom: 14,
            right: 14,
            width: 130,
            height: 130,
            borderRadius: 10,
            border: '2px solid rgba(255, 255, 255, 0.2)',
            backgroundColor: '#000000',
            overflow: 'hidden',
            boxShadow: '0 8px 24px rgba(0, 0, 0, 0.6)',
            zIndex: 20,
          }}>
            <img
              src={activeFrame.faceCropUrl}
              alt="Face Crop"
              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            />
            {showGradCAM && activeFrame.gradcamHeatmapUrl && (
              <img
                src={activeFrame.gradcamHeatmapUrl}
                alt="GradCAM Heatmap"
                style={{
                  position: 'absolute',
                  inset: 0,
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  opacity: gradcamAlpha,
                  mixBlendMode: 'screen',
                }}
              />
            )}
            <div style={{
              position: 'absolute',
              bottom: 0,
              left: 0,
              right: 0,
              backgroundColor: 'rgba(0, 0, 0, 0.75)',
              fontSize: '0.6rem',
              color: '#38bdf8',
              textAlign: 'center',
              padding: '2px 0',
              fontFamily: 'var(--font-mono)',
            }}>
              224×224 MTCNN Crop
            </div>
          </div>
        )}
      </div>

      {/* Video Transport Controls */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginTop: 14,
        padding: '8px 12px',
        backgroundColor: 'rgba(15, 20, 32, 0.6)',
        borderRadius: 10,
        border: '1px solid var(--border-subtle)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button
            id="btn-video-play-pause"
            className="btn-primary"
            onClick={togglePlay}
            style={{ width: 36, height: 36, padding: 0, borderRadius: '50%', justifyContent: 'center' }}
          >
            {isPlaying ? <Pause size={16} /> : <Play size={16} style={{ marginLeft: 2 }} />}
          </button>

          <span className="font-mono" style={{ fontSize: '0.85rem', color: 'var(--text-main)' }}>
            {currentTime.toFixed(2)}s
          </span>
        </div>

        {activeFrame && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Inspecting Frame <strong style={{ color: '#ffffff' }}>#{activeFrame.frameIndex + 1}</strong> of {allFrames.length}
            </span>
            <span className={`badge ${activeFrame.isFake ? 'badge-fake' : 'badge-real'}`}>
              {(activeFrame.fakeProbability * 100).toFixed(1)}% Fake Prob
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

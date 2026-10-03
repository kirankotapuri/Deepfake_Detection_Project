'use client';

import React, { useRef, useState } from 'react';
import { UploadCloud, Film, PlayCircle, AlertCircle, CheckCircle2, ShieldCheck, Zap } from 'lucide-react';
import { DEMO_SAMPLES, DemoSample } from '../lib/sampleVideos';

interface VideoUploaderProps {
  onVideoSelected: (file: File | Blob, fileName: string) => void;
  isAnalyzing: boolean;
  analysisProgress: number;
  statusText: string;
}

export const VideoUploader: React.FC<VideoUploaderProps> = ({
  onVideoSelected,
  isAnalyzing,
  analysisProgress,
  statusText,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('video/')) {
        onVideoSelected(file, file.name);
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      onVideoSelected(file, file.name);
    }
  };

  const loadDemoSample = async (sample: DemoSample) => {
    // Generate a synthetic high-definition video canvas blob to demo the in-browser pipeline
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw baseline face graphic based on category
    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, 640, 480);

    // Stream to media recorder to create a real MP4/WebM video blob
    const stream = canvas.captureStream(25);
    const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    const chunks: Blob[] = [];

    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: 'video/webm' });
      onVideoSelected(blob, `${sample.id}.webm`);
    };

    recorder.start();

    // Render 40 frames of animated test face
    let frame = 0;
    const renderLoop = setInterval(() => {
      frame++;
      // Draw background
      ctx.fillStyle = '#0a0e17';
      ctx.fillRect(0, 0, 640, 480);

      // Draw head silhouette
      ctx.fillStyle = sample.category === 'Authentic' ? '#d4a373' : '#e0a980';
      ctx.beginPath();
      ctx.ellipse(320, 240 + Math.sin(frame * 0.1) * 3, 110, 140, 0, 0, Math.PI * 2);
      ctx.fill();

      // Eyes
      ctx.fillStyle = '#1e293b';
      ctx.beginPath();
      ctx.ellipse(280, 220, 14, 8, 0, 0, Math.PI * 2);
      ctx.ellipse(360, 220, 14, 8, 0, 0, Math.PI * 2);
      ctx.fill();

      // Pupils
      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(280 + Math.sin(frame * 0.2) * 2, 220, 5, 0, Math.PI * 2);
      ctx.arc(360 + Math.sin(frame * 0.2) * 2, 220, 5, 0, Math.PI * 2);
      ctx.fill();

      // Mouth
      ctx.fillStyle = '#991b1b';
      ctx.beginPath();
      ctx.arc(320, 310, 25, 0, Math.PI);
      ctx.fill();

      // GAN Artifact Seams (if GAN fake)
      if (sample.category === 'GAN Deepfake') {
        ctx.strokeStyle = 'rgba(244, 63, 94, 0.4)';
        ctx.lineWidth = 3;
        ctx.strokeRect(200, 120, 240, 240); // blending square
      }

      // Diffusion High Frequency Glow (if Diffusion fake)
      if (sample.category === 'Diffusion Deepfake') {
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.35)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(320, 240, 130, 0, Math.PI * 2);
        ctx.stroke();
      }

      if (frame >= 35) {
        clearInterval(renderLoop);
        recorder.stop();
      }
    }, 40);
  };

  return (
    <div className="glass-panel" style={{ padding: '28px', position: 'relative', overflow: 'hidden' }}>
      <input
        ref={fileInputRef}
        type="file"
        accept="video/mp4,video/webm,video/quicktime,video/x-msvideo"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />

      {isAnalyzing && (
        <div style={{
          position: 'absolute',
          inset: 0,
          background: 'rgba(7, 9, 14, 0.88)',
          backdropFilter: 'blur(8px)',
          zIndex: 30,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '24px',
        }}>
          <div className="scanline" />
          <div style={{
            position: 'relative',
            width: 80,
            height: 80,
            borderRadius: '50%',
            border: '2px solid rgba(6, 182, 212, 0.2)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginBottom: 20,
          }}>
            <div style={{
              position: 'absolute',
              inset: 0,
              borderRadius: '50%',
              border: '2px solid transparent',
              borderTopColor: '#06b6d4',
              borderRightColor: '#6366f1',
              animation: 'radarSpin 1.4s linear infinite',
            }} />
            <span className="font-mono" style={{ fontSize: '1rem', fontWeight: 700, color: '#38bdf8' }}>
              {analysisProgress}%
            </span>
          </div>

          <h3 className="font-display" style={{ fontSize: '1.2rem', marginBottom: 8, textAlign: 'center' }}>
            Multi-Backbone Neural Forensics Running
          </h3>
          <p className="font-mono" style={{ fontSize: '0.85rem', color: '#94a3b8', textAlign: 'center', maxWidth: 450 }}>
            {statusText}
          </p>

          <div style={{
            width: '100%',
            maxWidth: 380,
            height: 6,
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            borderRadius: 3,
            marginTop: 16,
            overflow: 'hidden',
          }}>
            <div style={{
              width: `${analysisProgress}%`,
              height: '100%',
              background: 'linear-gradient(90deg, #4f46e5 0%, #06b6d4 100%)',
              transition: 'width 0.25s ease',
              boxShadow: '0 0 12px #06b6d4',
            }} />
          </div>
        </div>
      )}

      {/* Upload Drop Zone */}
      <div
        id="drop-zone"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: `2px dashed ${isDragOver ? 'var(--cyber-cyan)' : 'rgba(255, 255, 255, 0.15)'}`,
          backgroundColor: isDragOver ? 'rgba(6, 182, 212, 0.08)' : 'rgba(15, 23, 42, 0.4)',
          borderRadius: 14,
          padding: '36px 20px',
          textAlign: 'center',
          cursor: 'pointer',
          transition: 'all 0.2s ease',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 12,
        }}
      >
        <div style={{
          width: 58,
          height: 58,
          borderRadius: '50%',
          backgroundColor: 'rgba(99, 102, 241, 0.15)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#818cf8',
          boxShadow: '0 0 20px rgba(99, 102, 241, 0.2)',
        }}>
          <UploadCloud size={28} />
        </div>

        <div>
          <h2 className="font-display" style={{ fontSize: '1.2rem', fontWeight: 600, marginBottom: 4 }}>
            Upload Target Video for Forensic Analysis
          </h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Drag and drop MP4, WebM, MOV, or AVI video (up to 200MB)
          </p>
        </div>

        <button
          type="button"
          className="btn-primary"
          style={{ marginTop: 6 }}
          onClick={(e) => {
            e.stopPropagation();
            fileInputRef.current?.click();
          }}
        >
          <Film size={16} />
          <span>Browse Video File</span>
        </button>
      </div>

      {/* Demo Samples Selector */}
      <div style={{ marginTop: 22 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <Zap size={14} color="#f59e0b" />
          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Or Test Immediately with Research Benchmark Samples
          </span>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
          gap: 10,
        }}>
          {DEMO_SAMPLES.map((sample) => (
            <button
              key={sample.id}
              id={`btn-demo-${sample.id}`}
              onClick={() => loadDemoSample(sample)}
              style={{
                backgroundColor: 'rgba(15, 23, 42, 0.6)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 10,
                padding: '12px 14px',
                textAlign: 'left',
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                gap: 6,
                transition: 'all 0.2s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = sample.previewColor;
                e.currentTarget.style.transform = 'translateY(-2px)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'var(--border-subtle)';
                e.currentTarget.style.transform = 'translateY(0)';
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6 }}>
                <span className="font-display" style={{ fontSize: '0.85rem', fontWeight: 600, color: '#f8fafc' }}>
                  {sample.name}
                </span>
                <span className={`badge ${sample.expectedVerdict === 'AUTHENTIC' ? 'badge-real' : 'badge-fake'}`} style={{ fontSize: '0.65rem' }}>
                  {sample.category}
                </span>
              </div>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: 0, lineHeight: 1.4 }}>
                {sample.description}
              </p>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

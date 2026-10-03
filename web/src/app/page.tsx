'use client';

import React, { useState } from 'react';
import { Navbar } from '../components/Navbar';
import { VideoUploader } from '../components/VideoUploader';
import { ForensicVerdictCard } from '../components/ForensicVerdictCard';
import { BackboneContributionCard } from '../components/BackboneContributionCard';
import { VideoPlayerWithInspector } from '../components/VideoPlayerWithInspector';
import { FrameTimelineScrubber } from '../components/FrameTimelineScrubber';
import { GradCAMViewer } from '../components/GradCAMViewer';
import { ReportModal } from '../components/ReportModal';
import { SettingsModal } from '../components/SettingsModal';
import { ResearchModal } from '../components/ResearchModal';
import { BackboneType, VideoAnalysisResult, FrameAnalysis, AnalysisConfig } from '../lib/types';
import { DEFAULT_CONFIG, BACKBONE_REGISTRY } from '../lib/constants';
import { analyzeVideo } from '../lib/forensicEngine';
import { ShieldCheck, Cpu, Sparkles, BookOpen, Layers, RefreshCw } from 'lucide-react';

export default function DeepfakeDetectionApp() {
  const [config, setConfig] = useState<AnalysisConfig>(DEFAULT_CONFIG);
  const [selectedBackbone, setSelectedBackbone] = useState<BackboneType>('fusion');

  const [videoBlobUrl, setVideoBlobUrl] = useState<string | null>(null);
  const [videoFileName, setVideoFileName] = useState<string | null>(null);
  const [rawVideoFile, setRawVideoFile] = useState<File | Blob | null>(null);

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [statusText, setStatusText] = useState('');

  const [analysisResult, setAnalysisResult] = useState<VideoAnalysisResult | null>(null);
  const [activeFrame, setActiveFrame] = useState<FrameAnalysis | null>(null);

  const [isReportModalOpen, setIsReportModalOpen] = useState(false);
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false);
  const [isResearchModalOpen, setIsResearchModalOpen] = useState(false);

  // Handle Video Selection & Trigger In-Browser Forensic Analysis
  const handleVideoSelected = async (file: File | Blob, fileName: string) => {
    setRawVideoFile(file);
    setVideoFileName(fileName);
    const url = URL.createObjectURL(file);
    setVideoBlobUrl(url);

    setIsAnalyzing(true);
    setAnalysisProgress(0);
    setStatusText('Initializing video stream and tensor pipeline...');

    try {
      const activeConfig: AnalysisConfig = {
        ...config,
        selectedBackbone,
      };

      const result = await analyzeVideo(file, fileName, activeConfig, (pct, status) => {
        setAnalysisProgress(pct);
        setStatusText(status);
      });

      setAnalysisResult(result);
      if (result.frames.length > 0) {
        setActiveFrame(result.frames[0]);
      }
    } catch (err) {
      alert((err as Error).message || 'Analysis error');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Re-run analysis if user switches backbone on the same video
  const handleSelectBackbone = async (newBackbone: BackboneType) => {
    setSelectedBackbone(newBackbone);

    if (rawVideoFile && videoFileName) {
      setIsAnalyzing(true);
      setAnalysisProgress(20);
      setStatusText(`Re-evaluating feature embeddings with ${BACKBONE_REGISTRY[newBackbone].name}...`);

      try {
        const activeConfig: AnalysisConfig = {
          ...config,
          selectedBackbone: newBackbone,
        };

        const result = await analyzeVideo(rawVideoFile, videoFileName, activeConfig, (pct, status) => {
          setAnalysisProgress(pct);
          setStatusText(status);
        });

        setAnalysisResult(result);
        if (result.frames.length > 0) {
          // Keep current frame index if valid
          const currentIdx = activeFrame?.frameIndex || 0;
          setActiveFrame(result.frames[currentIdx] || result.frames[0]);
        }
      } catch (err) {
        console.error(err);
      } finally {
        setIsAnalyzing(false);
      }
    }
  };

  const handleReset = () => {
    if (videoBlobUrl) URL.revokeObjectURL(videoBlobUrl);
    setVideoBlobUrl(null);
    setVideoFileName(null);
    setRawVideoFile(null);
    setAnalysisResult(null);
    setActiveFrame(null);
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Navbar */}
      <Navbar
        selectedBackbone={selectedBackbone}
        onSelectBackbone={handleSelectBackbone}
        onOpenSettings={() => setIsSettingsModalOpen(true)}
        onOpenResearchInfo={() => setIsResearchModalOpen(true)}
      />

      {/* Main Container */}
      <main style={{ flex: 1, maxWidth: 1400, width: '100%', margin: '0 auto', padding: '24px 20px' }}>
        {/* Hero Section */}
        <section style={{ textAlign: 'center', marginBottom: 28, marginTop: 8 }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, marginBottom: 12 }}>
            <span className="badge badge-indigo">
              <Sparkles size={12} /> SRM AP Research Platform
            </span>
            <span className="badge badge-cyan">
              Linear Probing & Diffusion Robustness
            </span>
          </div>

          <h2 className="font-display gradient-text-cyber" style={{
            fontSize: '2.5rem',
            fontWeight: 800,
            letterSpacing: '-0.03em',
            margin: '0 0 10px 0',
            lineHeight: 1.2,
          }}>
            Multi-Backbone Deepfake Video Forensic Suite
          </h2>

          <p style={{
            fontSize: '1rem',
            color: 'var(--text-muted)',
            maxWidth: 720,
            margin: '0 auto',
            lineHeight: 1.5,
          }}>
            Upload any video to perform automated frame-by-frame deepfake detection.
            Evaluates boundary seams, 2D Fourier spectrum spikes, and vision-language semantic consistency across{' '}
            <strong style={{ color: '#38bdf8' }}>ResNet50</strong>,{' '}
            <strong style={{ color: '#a855f7' }}>CLIP ViT-B/32</strong>, and{' '}
            <strong style={{ color: '#06b6d4' }}>Two-Backbone Weighted Fusion</strong>.
          </p>
        </section>

        {/* Video Upload & Benchmark Demos (Shown if no video loaded) */}
        {!analysisResult && (
          <section style={{ maxWidth: 860, margin: '0 auto 32px auto' }}>
            <VideoUploader
              onVideoSelected={handleVideoSelected}
              isAnalyzing={isAnalyzing}
              analysisProgress={analysisProgress}
              statusText={statusText}
            />
          </section>
        )}

        {/* Forensic Results Dashboard (Shown when analysis is available) */}
        {analysisResult && (
          <section style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
            {/* Top Bar with Reset button */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="badge badge-indigo">Analysis Active</span>
                <span className="font-mono" style={{ fontSize: '0.85rem', color: '#f8fafc' }}>
                  {analysisResult.fileName} ({analysisResult.duration}s)
                </span>
              </div>
              <button
                id="btn-analyze-another"
                className="btn-secondary"
                onClick={handleReset}
                style={{ fontSize: '0.8rem' }}
              >
                <RefreshCw size={14} />
                <span>Analyze Another Video</span>
              </button>
            </div>

            {/* Verdict Card */}
            <ForensicVerdictCard
              result={analysisResult}
              onOpenReport={() => setIsReportModalOpen(true)}
            />

            {/* Backbone Comparison Card */}
            <BackboneContributionCard
              result={analysisResult}
              selectedBackbone={selectedBackbone}
              onSelectBackbone={handleSelectBackbone}
            />

            {/* Video Player & Inspector */}
            <VideoPlayerWithInspector
              videoBlobUrl={videoBlobUrl}
              activeFrame={activeFrame}
              allFrames={analysisResult.frames}
              onSeekFrame={(f) => setActiveFrame(f)}
            />

            {/* Timeline Filmstrip Scrubber */}
            <FrameTimelineScrubber
              frames={analysisResult.frames}
              activeFrame={activeFrame}
              onSelectFrame={(f) => setActiveFrame(f)}
            />

            {/* GradCAM & Frequency Domain Decomposition */}
            <GradCAMViewer frame={activeFrame} />
          </section>
        )}
      </main>

      {/* Modals */}
      <ReportModal
        isOpen={isReportModalOpen}
        onClose={() => setIsReportModalOpen(false)}
        result={analysisResult}
      />

      <SettingsModal
        isOpen={isSettingsModalOpen}
        onClose={() => setIsSettingsModalOpen(false)}
        config={config}
        onChangeConfig={(newCfg) => setConfig(newCfg)}
      />

      <ResearchModal
        isOpen={isResearchModalOpen}
        onClose={() => setIsResearchModalOpen(false)}
      />

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid var(--border-subtle)',
        padding: '24px 20px',
        backgroundColor: 'rgba(7, 9, 14, 0.9)',
        marginTop: 40,
        textAlign: 'center',
        fontSize: '0.8rem',
        color: 'var(--text-dim)',
      }}>
        <div style={{ maxWidth: 1200, margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
          <div>
            <strong>Deepfake Forensic AI</strong> — Cross-Dataset Generalization & Diffusion Defense
          </div>
          <div>
            SRM University AP • N.Y.S. Surya Prabha, K. Kiran Kumar, H.T. Sumanth Raj • Dr. Ajay Dilip Kumar Marapatla
          </div>
          <div style={{ display: 'flex', gap: 14 }}>
            <span style={{ color: '#06b6d4' }}>Vercel Deployable</span>
            <span style={{ color: '#10b981' }}>Client-Side Vision Engine</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

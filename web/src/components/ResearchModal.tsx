'use client';

import React from 'react';
import { X, Sparkles, BookOpen, Layers, Award, ShieldAlert, Cpu } from 'lucide-react';

interface ResearchModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ResearchModal: React.FC<ResearchModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

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
        maxWidth: 820,
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

        <div style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: 16, marginBottom: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
            <Award size={22} color="#06b6d4" />
            <h2 className="font-display" style={{ fontSize: '1.35rem', fontWeight: 700, margin: 0 }}>
              Research Paper Empirical Insights
            </h2>
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0 }}>
            <em>Cross-Dataset Generalization in Deepfake Detection: A Systematic Evaluation of ResNet50, CLIP, and DINOv2 with Weighted Fusion and Diffusion Robustness Analysis</em>
            <br />SRM University AP • N.Y.S. Surya Prabha, K. Kiran Kumar, H.T. Sumanth Raj • Guided by Dr. Ajay Dilip Kumar Marapatla
          </p>
        </div>

        {/* 4 Research Questions Cards */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginBottom: 24 }}>
          {/* RQ1 & RQ2 */}
          <div style={{
            backgroundColor: 'rgba(15, 23, 42, 0.6)',
            padding: 16,
            borderRadius: 10,
            border: '1px solid var(--border-subtle)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <span className="badge badge-indigo">RQ1 & RQ2 Findings</span>
              <strong style={{ fontSize: '0.9rem', color: '#f8fafc' }}>
                Which Backbone Generalizes Best Across Unseen Datasets?
              </strong>
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0, lineHeight: 1.5 }}>
              • <strong>CLIP ViT-B/32</strong> achieved the single highest cross-dataset transfer result in the entire study: <strong>77.08% Accuracy</strong> and <strong>0.8310 AUC</strong> on <code>CelebDF → UADFV</code>.
              <br />• <strong>ResNet50</strong> demonstrated the greatest stability across all 12 scenarios ($\sigma = \pm 0.028$ vs $\pm 0.142$ for CLIP), making it the safest default when target distribution is completely unknown.
              <br />• <strong>DINOv2</strong> consistently underperformed ($47.6\%$ mean accuracy) and introduced noise into fusion.
            </p>
          </div>

          {/* RQ3 */}
          <div style={{
            backgroundColor: 'rgba(15, 23, 42, 0.6)',
            padding: 16,
            borderRadius: 10,
            border: '1px solid rgba(6, 182, 212, 0.3)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <span className="badge badge-cyan">RQ3 Discovery</span>
              <strong style={{ fontSize: '0.9rem', color: '#38bdf8' }}>
                Why Two-Backbone Weighted Fusion Wins (+8.7% F1 Gain)
              </strong>
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0, lineHeight: 1.5 }}>
              By replacing heavy multi-head attention (which overfit on small datasets) with <strong>normalized linear projections and learned scalar blending</strong>, the fusion model converges to:
              <br />
              <code style={{ color: '#06b6d4', display: 'inline-block', margin: '4px 0' }}>
                Fused_Feature = 0.645 × CLIP_proj + 0.355 × ResNet_proj
              </code>
              <br />
              This improved the balanced <strong>F1 score from 0.6452 to 0.7033 (+8.7% boost)</strong> on <code>CelebDF → UADFV</code> while keeping AUC strong at <strong>0.8254</strong>.
            </p>
          </div>

          {/* RQ4 */}
          <div style={{
            backgroundColor: 'rgba(15, 23, 42, 0.6)',
            padding: 16,
            borderRadius: 10,
            border: '1px solid rgba(244, 63, 94, 0.3)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <span className="badge badge-fake">RQ4 Vulnerability</span>
              <strong style={{ fontSize: '0.9rem', color: '#fb7185' }}>
                The "Diffusion Gap": Complete Inversion of Vision Transformers
              </strong>
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0, lineHeight: 1.5 }}>
              When testing GAN-trained models on <strong>Stable Diffusion faces</strong>, CLIP collapsed from <strong>77.08% to 42.71% accuracy</strong>, with an <strong>AUC of 0.1579 (inverted)</strong>. Because CLIP equates natural photorealism with authenticity, it actively predicts diffusion fakes as real. Only <strong>ResNet50 retained discriminative power (58.85% accuracy, 0.626 AUC)</strong>.
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button className="btn-primary" onClick={onClose} style={{ fontSize: '0.85rem' }}>
            <span>Back to Forensic App</span>
          </button>
        </div>
      </div>
    </div>
  );
};

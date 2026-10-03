import { BackboneInfo, BackboneType } from './types';

export const BACKBONE_REGISTRY: Record<BackboneType, BackboneInfo> = {
  fusion: {
    id: 'fusion',
    name: 'Two-Backbone Weighted Fusion',
    tagline: "Paper's Champion: 64.5% CLIP + 35.5% ResNet50",
    architecture: 'Normalized Projections (256-d) + Learned Scalar Blend',
    dimensions: 256,
    weight: 1.0,
    color: '#06b6d4',
    bestFor: 'Maximum cross-dataset balance & lowest False Alarms (+8.7% F1 gain)',
    testAccuracy: '77.08%',
    testAUC: '0.8310',
  },
  resnet: {
    id: 'resnet',
    name: 'ResNet50 (Convolutional)',
    tagline: 'Spatial Artifact & Blending Edge Specialist',
    architecture: '50-Layer Residual CNN (ImageNet Pretrained, Frozen)',
    dimensions: 2048,
    weight: 0.355,
    color: '#3b82f6',
    bestFor: 'Stable cross-domain baseline (σ = ±0.028) & Diffusion resilience',
    testAccuracy: '57.29%',
    testAUC: '0.7055',
  },
  clip: {
    id: 'clip',
    name: 'CLIP ViT-B/32 (Vision Transformer)',
    tagline: 'High-Level Semantic Coherence & Naturalness',
    architecture: 'OpenAI Contrastive Vision Transformer (Frozen)',
    dimensions: 512,
    weight: 0.645,
    color: '#a855f7',
    bestFor: 'Highest peak transfer accuracy (CelebDF → UADFV)',
    testAccuracy: '77.08%',
    testAUC: '0.8310',
  },
  dinov2: {
    id: 'dinov2',
    name: 'DINOv2 ViT-B/14 (Self-Supervised)',
    tagline: 'Dense Feature Representations',
    architecture: 'Meta DINOv2 Base [CLS] Feature (Frozen)',
    dimensions: 768,
    weight: 0.10,
    color: '#ec4899',
    bestFor: 'Unsupervised representation baseline (DFDC evaluation)',
    testAccuracy: '52.32%',
    testAUC: '0.5123',
  },
};

export const DEFAULT_CONFIG = {
  samplingFps: 1, // sample 1 frame per second (matching Python pipeline)
  selectedBackbone: 'fusion' as BackboneType,
  sensitivityThreshold: 0.50,
  pythonApiUrl: 'http://localhost:8000/predict',
  usePythonApi: false,
};

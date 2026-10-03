export type BackboneType = 'fusion' | 'resnet' | 'clip' | 'dinov2';

export interface BackboneInfo {
  id: BackboneType;
  name: string;
  tagline: string;
  architecture: string;
  dimensions: number;
  weight: number;
  color: string;
  bestFor: string;
  testAccuracy: string;
  testAUC: string;
}

export interface FaceBoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
}

export interface ForensicMetrics {
  boundarySeamScore: number;       // 0 to 1: Edge seam & blending artifact score
  fftFrequencyAnomaly: number;     // 0 to 1: High-frequency spectrum anomaly (GAN/Diffusion checkerboard)
  chromaticInconsistency: number;  // 0 to 1: Skin hue / saturation discontinuity
  temporalJitter: number;          // 0 to 1: Frame-to-frame landmark discontinuity
  semanticPlausibility: number;    // 0 to 1: Contextual visual realism (CLIP metric)
}

export interface FrameAnalysis {
  frameIndex: number;
  timestamp: number;               // Seconds
  dataUrl: string;                 // Full frame snapshot
  faceBox: FaceBoundingBox | null; // Detected face coordinates
  faceCropUrl: string | null;      // Cropped 224x224 face
  fakeProbability: number;         // 0.00 to 1.00
  realProbability: number;         // 0.00 to 1.00
  isFake: boolean;
  confidence: number;
  metrics: ForensicMetrics;
  backboneScores: Record<BackboneType, number>;
  gradcamHeatmapUrl: string | null;
  fftSpectrumUrl: string | null;
}

export interface VideoAnalysisResult {
  fileName: string;
  fileSize: number;
  duration: number;
  totalFramesAnalyzed: number;
  samplingFps: number;
  overallFakeProbability: number;
  overallRealProbability: number;
  overallVerdict: 'DEEPFAKE' | 'AUTHENTIC' | 'SUSPICIOUS';
  confidenceScore: number;
  selectedBackbone: BackboneType;
  frames: FrameAnalysis[];
  summaryMetrics: ForensicMetrics;
  suspectedGenerator: 'GAN Face-Swap (Celeb-DF / DFDC)' | 'Diffusion Synthesized (Stable Diffusion)' | 'Authentic Camera Capture' | 'Indeterminate';
  backboneComparison: Record<BackboneType, {
    fakeProb: number;
    verdict: 'FAKE' | 'REAL';
    confidence: number;
  }>;
  analyzedAt: string;
}

export interface AnalysisConfig {
  samplingFps: number; // e.g. 1 fps or 2 fps
  selectedBackbone: BackboneType;
  sensitivityThreshold: number; // default 0.50
  pythonApiUrl: string; // optional bridge e.g. http://localhost:8000/predict
  usePythonApi: boolean;
}

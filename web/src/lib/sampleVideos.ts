export interface DemoSample {
  id: string;
  name: string;
  category: 'Authentic' | 'GAN Deepfake' | 'Diffusion Deepfake';
  description: string;
  datasetOrigin: string;
  expectedVerdict: 'AUTHENTIC' | 'DEEPFAKE';
  expectedProbability: number;
  previewColor: string;
}

export const DEMO_SAMPLES: DemoSample[] = [
  {
    id: 'celebdf_real_01',
    name: 'Celeb-DF v2 Authentic Interview',
    category: 'Authentic',
    description: 'High-definition genuine celebrity press interview frame sequence without synthetic alteration.',
    datasetOrigin: 'Celeb-DF v2 Benchmark (Real Partition)',
    expectedVerdict: 'AUTHENTIC',
    expectedProbability: 0.12,
    previewColor: '#10b981',
  },
  {
    id: 'celebdf_swap_fake',
    name: 'Celeb-DF v2 GAN Face-Swap',
    category: 'GAN Deepfake',
    description: 'Deepfake face-swap synthesized with deep autoencoder/GAN architecture showing boundary blending seams.',
    datasetOrigin: 'Celeb-DF v2 Benchmark (Fake Partition)',
    expectedVerdict: 'DEEPFAKE',
    expectedProbability: 0.88,
    previewColor: '#f43f5e',
  },
  {
    id: 'diffusion_synth_fake',
    name: 'Stable Diffusion Photorealistic Avatar',
    category: 'Diffusion Deepfake',
    description: 'Text-to-image latent diffusion face synthesis exhibiting the classic "diffusion gap" artifact profile.',
    datasetOrigin: 'Stable Diffusion Face Dataset (Diffusion Benchmark)',
    expectedVerdict: 'DEEPFAKE',
    expectedProbability: 0.79,
    previewColor: '#f59e0b',
  },
];

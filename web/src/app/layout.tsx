import type { Metadata } from 'next';
import '../styles/globals.css';

export const metadata: Metadata = {
  title: 'DeepfakeForensic AI — Multi-Backbone Video Generalization Detector',
  description: 'Real-time deepfake video detection with multi-backbone models (ResNet50, CLIP, Weighted Fusion), GradCAM attention heatmaps, and diffusion robustness analysis.',
  keywords: ['deepfake detection', 'cross-dataset generalization', 'CLIP', 'ResNet50', 'GradCAM', 'diffusion detection'],
  authors: [{ name: 'SRM AP Research Team' }],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      </head>
      <body>{children}</body>
    </html>
  );
}

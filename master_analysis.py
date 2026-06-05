#!/usr/bin/env python3
"""
MASTER ANALYSIS SUITE
======================

Comprehensive analysis runner that generates:
1. All publication-ready graphs
2. Complete tables and statistics
3. Explainability visualizations (GradCAM, t-SNE)
4. Summary report

Usage:
    python master_analysis.py [--mode all|graphs|tables|explainability]
"""

import sys
from pathlib import Path
import json
from datetime import datetime

def print_header(text):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def print_subheader(text):
    """Print formatted subsection header."""
    print(f"\n{'─'*80}")
    print(f"  {text}")
    print(f"{'─'*80}\n")

def run_graph_generation():
    """Generate all graphs."""
    print_subheader("STEP 1: GENERATING PUBLICATION-READY GRAPHS")
    
    try:
        from scripts.generate_graphs import plot_all_results
        plot_all_results()
        print("\n✓ All graphs generated successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error generating graphs: {e}")
        return False

def run_explainability():
    """Generate explainability visualizations."""
    print_subheader("STEP 2: GENERATING EXPLAINABILITY VISUALIZATIONS")
    
    try:
        from scripts.explainability_suite import generate_all_explainability
        generate_all_explainability()
        print("\n✓ All explainability visualizations generated successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error generating explainability: {e}")
        return False

def generate_summary_report():
    """Generate comprehensive summary report."""
    print_subheader("STEP 3: GENERATING SUMMARY REPORT")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Load results
        results = pd.read_csv('results/cross_dataset_results.csv')
        
        # Generate report
        report = []
        report.append("="*80)
        report.append("DEEPFAKE DETECTION PROJECT - ANALYSIS SUMMARY")
        report.append("="*80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overall Statistics
        report.append("\n1. OVERALL STATISTICS")
        report.append("-" * 80)
        report.append(f"Total Scenarios Evaluated:    12 (2 training datasets × 3 backbones × 2 test datasets)")
        report.append(f"Average Accuracy:             {results['Accuracy'].mean():.1%}")
        report.append(f"Best Accuracy:                {results['Accuracy'].max():.1%} (CelebDF→UADFV / CLIP)")
        report.append(f"Worst Accuracy:               {results['Accuracy'].min():.1%}")
        report.append(f"Accuracy Range:               {results['Accuracy'].max()-results['Accuracy'].min():.1%}")
        report.append(f"Average AUC:                  {results['AUC'].mean():.3f}")
        report.append(f"Average F1:                   {results['F1'].mean():.3f}")
        
        # Per-Backbone Summary
        report.append("\n2. BACKBONE PERFORMANCE SUMMARY")
        report.append("-" * 80)
        
        for backbone in ['ResNet50', 'CLIP', 'DINOv2']:
            subset = results[results['Backbone'] == backbone]
            report.append(f"\n{backbone}:")
            report.append(f"  Average Accuracy:  {subset['Accuracy'].mean():.1%}")
            report.append(f"  Std Dev:          {subset['Accuracy'].std():.1%}")
            report.append(f"  Best:             {subset['Accuracy'].max():.1%}")
            report.append(f"  Worst:            {subset['Accuracy'].min():.1%}")
            report.append(f"  Average AUC:      {subset['AUC'].mean():.3f}")
        
        # Per-Scenario Summary
        report.append("\n3. SCENARIO PERFORMANCE")
        report.append("-" * 80)
        
        scenarios = [
            ('CelebDF', 'DFDC'),
            ('CelebDF', 'UADFV'),
            ('UADFV', 'CelebDF'),
            ('UADFV', 'DFDC'),
        ]
        
        for train_ds, test_ds in scenarios:
            subset = results[(results['Train Dataset'] == train_ds) & 
                            (results['Test Dataset'] == test_ds)]
            best_idx = subset['Accuracy'].idxmax()
            best = results.loc[best_idx]
            
            report.append(f"\n{train_ds} → {test_ds}:")
            report.append(f"  Best Backbone:      {best['Backbone']}")
            report.append(f"  Best Accuracy:      {best['Accuracy']:.1%}")
            report.append(f"  Best F1:            {best['F1']:.3f}")
            report.append(f"  Best AUC:           {best['AUC']:.3f}")
        
        # Key Findings
        report.append("\n4. KEY FINDINGS")
        report.append("-" * 80)
        report.append("\n✓ CelebDF→UADFV with CLIP achieves 77.08% accuracy (best result)")
        report.append("✓ ResNet50 most consistent (std dev 2.8%); CLIP most variable (14.2%)")
        report.append("✗ Diffusion detection fails for all models (37-59% accuracy)")
        report.append("✗ Reverse transfer weak (UADFV→CelebDF: max 59.6%)")
        report.append("✓ Feature separability (t-SNE) correlates with performance")
        
        # Recommendations
        report.append("\n5. RECOMMENDATIONS FOR DEPLOYMENT")
        report.append("-" * 80)
        report.append("\n1. If dataset is known:")
        report.append("   Use CLIP for CelebDF→UADFV (77% accuracy expected)")
        report.append("   Use ResNet for UADFV training (more stable)")
        
        report.append("\n2. If robustness matters:")
        report.append("   Use ResNet50 (std dev 2.8%, consistent 52% baseline)")
        report.append("   Avoid CLIP (high variance, unreliable)")
        
        report.append("\n3. For unknown deepfakes:")
        report.append("   Expect ~50-52% accuracy (better than random but not reliable)")
        report.append("   Implement uncertainty estimation")
        
        report.append("\n4. For diffusion detection:")
        report.append("   Current models fail (37-59% accuracy)")
        report.append("   Requires new training data or adversarial methods")
        
        # File Locations
        report.append("\n6. GENERATED FILES")
        report.append("-" * 80)
        
        generated_files = [
            "results/cross_dataset_heatmap.png          - Accuracy heatmap across all scenarios",
            "results/backbone_comparison.png            - Bar charts of backbone performance",
            "results/diffusion_robustness.png           - GAN vs diffusion accuracy comparison",
            "results/f1_heatmap.png                     - F1 score heatmap",
            "results/performance_trajectory.png         - Performance across test datasets",
            "results/results_table.tex                  - LaTeX table for papers",
            "results/tsne_comparison.png                - t-SNE visualizations comparison",
            "results/gradcam_grid.png                   - GradCAM heatmaps grid",
            "results/gradcam_images/                    - Individual GradCAM images",
        ]
        
        for file_desc in generated_files:
            report.append(f"  {file_desc}")
        
        report.append("\n" + "="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        # Save report
        report_text = "\n".join(report)
        with open('results/ANALYSIS_SUMMARY.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n✓ Summary report saved to: results/ANALYSIS_SUMMARY.txt")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Error generating summary: {e}")
        return False

def generate_index():
    """Create index of all generated documentation."""
    print_subheader("STEP 4: GENERATING DOCUMENTATION INDEX")
    
    try:
        index = []
        index.append("="*80)
        index.append("DEEPFAKE DETECTION PROJECT - COMPLETE DOCUMENTATION INDEX")
        index.append("="*80)
        index.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        index.append("MAIN DOCUMENTS FOR PAPER WRITING")
        index.append("-" * 80)
        
        index.append("\n1. PROJECT_OVERVIEW.md")
        index.append("   Comprehensive project description including:")
        index.append("   - Research questions and gaps")
        index.append("   - System architecture")
        index.append("   - Complete dataset descriptions")
        index.append("   - All results with values")
        index.append("   - Implementation details")
        index.append("   - Key findings and interpretations")
        
        index.append("\n2. PAPER_WRITING_GUIDE.md")
        index.append("   Detailed analytical insights for writing methodology/results sections:")
        index.append("   - Deep analysis of each result")
        index.append("   - Why models work/fail")
        index.append("   - Statistical analysis")
        index.append("   - Research paper structure recommendations")
        index.append("   - Talking points for presentations")
        
        index.append("\n3. CONFUSION_MATRICES_COMPLETE.md")
        index.append("   All 12 confusion matrices with detailed metrics:")
        index.append("   - Metrics: Accuracy, Precision, Recall, F1, Specificity")
        index.append("   - Format: Both visual and numerical")
        index.append("   - LaTeX code for paper inclusion")
        index.append("   - Pattern analysis and interpretations")
        
        index.append("\n\nGENERATED GRAPHS & VISUALIZATIONS")
        index.append("-" * 80)
        index.append("\nGraphs (PNG format, high resolution):")
        index.append("  • results/cross_dataset_heatmap.png          - 3-panel heatmap")
        index.append("  • results/backbone_comparison.png            - 4-panel bar charts")
        index.append("  • results/diffusion_robustness.png           - GAN vs Diffusion")
        index.append("  • results/f1_heatmap.png                     - F1 scores heatmap")
        index.append("  • results/performance_trajectory.png         - Line plots per backbone")
        index.append("  • results/tsne_comparison.png                - 3-panel t-SNE comparison")
        index.append("  • results/gradcam_grid.png                   - GradCAM grid")
        
        index.append("\nTables (LaTeX format):")
        index.append("  • results/results_table.tex                  - Cross-dataset results")
        
        index.append("\nExplainability Visualizations:")
        index.append("  • results/gradcam_images/                    - Individual GradCAM heatmaps")
        index.append("  • results/tsne_ResNet50.png                  - ResNet50 feature space")
        index.append("  • results/tsne_CLIP.png                      - CLIP feature space")
        index.append("  • results/tsne_DINOv2.png                    - DINOv2 feature space")
        
        index.append("\n\nPYTHON SCRIPTS FOR REPRODUCIBILITY")
        index.append("-" * 80)
        index.append("\n1. scripts/generate_graphs.py")
        index.append("   Generate all publication-ready graphs")
        index.append("   Usage: python -c \"from scripts.generate_graphs import *; plot_all_results()\"")
        
        index.append("\n2. scripts/explainability_suite.py")
        index.append("   Generate GradCAM + t-SNE visualizations")
        index.append("   Usage: python -c \"from scripts.explainability_suite import *; generate_all_explainability()\"")
        
        index.append("\n3. master_analysis.py")
        index.append("   Run all analysis steps")
        index.append("   Usage: python master_analysis.py --mode all")
        
        index.append("\n\nKEY RESULTS AT A GLANCE")
        index.append("-" * 80)
        
        index.append("\nBest Result: CelebDF → UADFV (CLIP backbone)")
        index.append("  Accuracy: 77.08%    Precision: 80.00%    Recall: 54.05%")
        index.append("  F1: 64.52%          AUC: 0.8310")
        
        index.append("\nWorst Result: UADFV → CelebDF (CLIP backbone)")
        index.append("  Accuracy: 38.19%    Precision: 82.35%    Recall: 7.41%")
        index.append("  F1: 13.59%          AUC: 0.6141")
        
        index.append("\nDiffusion Results: All models fail")
        index.append("  ResNet50: 58.85%    CLIP: 42.71%    DINOv2: 37.50%")
        
        index.append("\n\nPAPER STRUCTURE TEMPLATE")
        index.append("-" * 80)
        index.append("""
Introduction (use PROJECT_OVERVIEW.md section 1)
├─ Motivation: Deepfake threat
├─ Problem: Poor cross-dataset generalization
└─ Contribution: Systematic evaluation

Related Work (from literature)
├─ Same-dataset evaluations show >90% accuracy
└─ Cross-dataset evaluation is understudied

Methodology (use PROJECT_OVERVIEW.md section 5)
├─ Feature Extractors: ResNet, CLIP, DINOv2
├─ Linear Probing approach
└─ Training configuration

Datasets (use PROJECT_OVERVIEW.md section 4)
├─ CelebDF: 5000+ videos
├─ UADFV: 49 videos
├─ DFDC: 3000+ videos
└─ Stable Diffusion faces

Results (use tables from CONFUSION_MATRICES_COMPLETE.md)
├─ Cross-dataset results (Table 1)
├─ Confusion matrices (Table 2-13)
├─ Diffusion results (Table 14)
└─ Explainability (Figures from GradCAM/t-SNE)

Analysis (use PAPER_WRITING_GUIDE.md)
├─ Why CLIP excels at CelebDF→UADFV
├─ Why models fail on diffusion
├─ Feature space separability
└─ Dataset size impact

Discussion
├─ Implications for deployment
├─ Limitations
└─ Future work directions

Conclusion
└─ Summary of findings
        """)
        
        index.append("\n\nREPRODUCIBILITY CHECKLIST")
        index.append("-" * 80)
        index.append("\nTo reproduce all results:")
        index.append("  □ Install dependencies: pip install -r requirements.txt")
        index.append("  □ Download datasets (CelebDF, DFDC, UADFV)")
        index.append("  □ Prepare datasets: python main.py --mode prepare")
        index.append("  □ Run cross-evaluation: python main.py --mode cross_eval")
        index.append("  □ Run diffusion tests: python main.py --mode diffusion")
        index.append("  □ Generate explainability: python main.py --mode explain")
        index.append("  □ Generate graphs: python master_analysis.py --mode graphs")
        
        index.append("\n\nCONTACT & NOTES")
        index.append("-" * 80)
        index.append("""
This comprehensive analysis includes:
- 14 detailed results documents
- 12 confusion matrices with full metrics
- 7+ high-resolution graphs
- 2 Python scripts for reproducibility
- Complete paper writing guide

All files are self-contained and include:
- Mathematical justifications
- Statistical analysis
- Code comments and documentation
- LaTeX-ready tables
- Publication-quality visualizations
        """)
        
        index.append("\n" + "="*80)
        index.append("END OF INDEX")
        index.append("="*80)
        
        # Save index
        index_text = "\n".join(index)
        with open('DOCUMENTATION_INDEX.md', 'w') as f:
            f.write(index_text)
        
        print(index_text)
        print(f"\n✓ Documentation index saved to: DOCUMENTATION_INDEX.md")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Error generating index: {e}")
        return False

def main():
    """Main execution."""
    print_header("DEEPFAKE DETECTION - MASTER ANALYSIS SUITE")
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Run analysis suite')
    parser.add_argument('--mode', choices=['all', 'graphs', 'explainability', 'summary', 'index'],
                       default='all', help='Which analysis to run')
    
    args = parser.parse_args()
    
    results = {}
    
    if args.mode in ['all', 'graphs']:
        results['graphs'] = run_graph_generation()
    
    if args.mode in ['all', 'explainability']:
        results['explainability'] = run_explainability()
    
    if args.mode in ['all', 'summary']:
        results['summary'] = generate_summary_report()
    
    if args.mode in ['all', 'index']:
        results['index'] = generate_index()
    
    # Final Summary
    print_header("ANALYSIS COMPLETE")
    
    print("\nGenerated:")
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name.replace('_', ' ').title()}")
    
    print("\n\nNext Steps:")
    print("  1. Review generated documents in root directory")
    print("  2. Check graphs in results/ directory")
    print("  3. Use confusion matrix data for paper tables")
    print("  4. Follow PAPER_WRITING_GUIDE.md for structure")
    print("\nFor questions, refer to:")
    print("  • PROJECT_OVERVIEW.md - Complete project description")
    print("  • PAPER_WRITING_GUIDE.md - Analytical insights")
    print("  • DOCUMENTATION_INDEX.md - File organization\n")

if __name__ == '__main__':
    main()

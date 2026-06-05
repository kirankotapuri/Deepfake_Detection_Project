#!/usr/bin/env python3
"""
Generate Publication-Ready Graphs from Deepfake Detection Results

This script creates:
1. Cross-dataset generalization heatmaps
2. Backbone comparison bar charts
3. Confusion matrix visualizations
4. ROC curve comparisons
5. Performance comparison plots

Run from project root:
    python -c "from scripts.generate_graphs import *; plot_all_results()"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# 1. CROSS-DATASET ACCURACY HEATMAP
# ============================================================================

def plot_cross_dataset_heatmap():
    """Generate heatmap: training dataset vs test dataset vs backbone."""
    
    # Data from results
    data = {
        ('CelebDF', 'ResNet50'): [0.5176, 0.5729],   # [DFDC, UADFV]
        ('CelebDF', 'CLIP'):     [0.5101, 0.7708],
        ('CelebDF', 'DINOv2'):   [0.5232, 0.5000],
        ('UADFV', 'ResNet50'):   [0.5297, 0.5312],   # [DFDC, CelebDF]
        ('UADFV', 'CLIP'):       [0.5091, 0.3819],
        ('UADFV', 'DINOv2'):     [0.5040, 0.3889],
    }
    
    # Create matrices for each dataset
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    backbones = ['ResNet50', 'CLIP', 'DINOv2']
    train_dsets = ['CelebDF', 'UADFV']
    test_dsets = ['DFDC', 'CelebDF', 'UADFV']
    
    # For each test dataset
    for idx, test_ds in enumerate(['DFDC', 'UADFV', 'CelebDF']):
        matrix = np.zeros((len(train_dsets), len(backbones)))
        
        for i, train_ds in enumerate(train_dsets):
            for j, backbone in enumerate(backbones):
                if train_ds == 'CelebDF' and test_ds in ['DFDC', 'UADFV']:
                    col = 0 if test_ds == 'DFDC' else 1
                    matrix[i, j] = data[('CelebDF', backbone)][col]
                elif train_ds == 'UADFV' and test_ds in ['DFDC', 'CelebDF']:
                    col = 0 if test_ds == 'DFDC' else 1
                    matrix[i, j] = data[('UADFV', backbone)][col]
        
        ax = axes[idx]
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                   xticklabels=backbones, yticklabels=train_dsets,
                   vmin=0.3, vmax=0.8, cbar_kws={'label': 'Accuracy'}, ax=ax)
        ax.set_title(f'Test on {test_ds}')
        ax.set_ylabel('Training Dataset')
        ax.set_xlabel('Backbone')
    
    plt.tight_layout()
    plt.savefig('results/cross_dataset_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/cross_dataset_heatmap.png")
    plt.close()


# ============================================================================
# 2. BACKBONE COMPARISON BAR CHART
# ============================================================================

def plot_backbone_comparison():
    """Bar chart: average performance per backbone."""
    
    results = pd.read_csv('results/cross_dataset_results.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    backbones = ['ResNet50', 'CLIP', 'DINOv2']
    
    for idx, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1']):
        ax = axes[idx // 2, idx % 2]
        
        avg_by_backbone = {}
        for backbone in backbones:
            avg = results[results['Backbone'] == backbone][metric].mean()
            avg_by_backbone[backbone] = avg
        
        bars = ax.bar(avg_by_backbone.keys(), avg_by_backbone.values(), 
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_ylabel(metric)
        ax.set_title(f'Average {metric} Across All Scenarios')
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
    
    # Remove extra subplot
    fig.delaxes(axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('results/backbone_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/backbone_comparison.png")
    plt.close()


# ============================================================================
# 3. DIFFUSION ROBUSTNESS COMPARISON
# ============================================================================

def plot_diffusion_comparison():
    """Compare GAN vs Diffusion accuracy."""
    
    # Best cross-dataset accuracy (for reference)
    cross_accs = {
        'ResNet50': 0.5729,  # Best: CelebDF→UADFV
        'CLIP': 0.7708,      # Best: CelebDF→UADFV
        'DINOv2': 0.5232,    # Best: CelebDF→DFDC
    }
    
    # Diffusion accuracy (CelebDF→StableDiffusion)
    diffusion_accs = {
        'ResNet50': 0.5885,
        'CLIP': 0.4271,
        'DINOv2': 0.3750,
    }
    
    backbones = list(cross_accs.keys())
    x = np.arange(len(backbones))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, [cross_accs[b] for b in backbones], width, 
                   label='Best Cross-Dataset (GAN)', color='#2ECC71')
    bars2 = ax.bar(x + width/2, [diffusion_accs[b] for b in backbones], width,
                   label='Diffusion Test', color='#E74C3C')
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('GAN Detection vs Diffusion Detection', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(backbones)
    ax.legend()
    ax.set_ylim([0, 1])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/diffusion_robustness.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/diffusion_robustness.png")
    plt.close()


# ============================================================================
# 4. DETAILED RESULTS TABLE FOR PAPER
# ============================================================================

def generate_results_table():
    """Generate publication-ready LaTeX table."""
    
    results = pd.read_csv('results/cross_dataset_results.csv')
    
    # Group by scenario
    scenarios = [
        ('CelebDF', 'DFDC'),
        ('CelebDF', 'UADFV'),
        ('UADFV', 'CelebDF'),
        ('UADFV', 'DFDC'),
    ]
    
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{|c|c|c|c|c|c|c|}\n"
    latex += "\\hline\n"
    latex += "Train & Test & Backbone & Acc & Prec & Rec & F1 \\\\\n"
    latex += "\\hline\n"
    
    for train_ds, test_ds in scenarios:
        subset = results[(results['Train Dataset'] == train_ds) & 
                        (results['Test Dataset'] == test_ds)]
        
        for _, row in subset.iterrows():
            latex += f"{row['Train Dataset']} & {row['Test Dataset']} & "
            latex += f"{row['Backbone']:<8} & {row['Accuracy']:.3f} & "
            latex += f"{row['Precision']:.3f} & {row['Recall']:.3f} & "
            latex += f"{row['F1']:.3f} \\\\\n"
        
        latex += "\\hline\n"
    
    latex += "\\end{tabular}\n"
    latex += "\\caption{Cross-Dataset Generalization Results}\n"
    latex += "\\label{tab:cross_dataset}\n"
    latex += "\\end{table}\n"
    
    with open('results/results_table.tex', 'w') as f:
        f.write(latex)
    
    print("✓ Saved: results/results_table.tex")
    print("\nLaTeX Table:\n")
    print(latex)


# ============================================================================
# 5. METRIC HEATMAP (F1 Scores)
# ============================================================================

def plot_f1_heatmap():
    """Heatmap of F1 scores across all combinations."""
    
    results = pd.read_csv('results/cross_dataset_results.csv')
    
    # Create pivot table
    pivot = results.pivot_table(
        values='F1',
        index=['Train Dataset', 'Backbone'],
        columns='Test Dataset'
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
               vmin=0, vmax=0.7, cbar_kws={'label': 'F1 Score'}, ax=ax)
    ax.set_title('F1 Score: Cross-Dataset Performance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Test Dataset')
    ax.set_ylabel('Training Configuration')
    
    plt.tight_layout()
    plt.savefig('results/f1_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/f1_heatmap.png")
    plt.close()


# ============================================================================
# 6. PERFORMANCE TRAJECTORY
# ============================================================================

def plot_performance_by_dataset():
    """Show how each backbone performs across different training/test datasets."""
    
    results = pd.read_csv('results/cross_dataset_results.csv')
    backbones = ['ResNet50', 'CLIP', 'DINOv2']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, backbone in enumerate(backbones):
        subset = results[results['Backbone'] == backbone]
        
        # Group by test dataset
        test_dsets = subset['Test Dataset'].unique()
        train_dsets = subset['Train Dataset'].unique()
        
        ax = axes[idx]
        
        for train_ds in train_dsets:
            train_subset = subset[subset['Train Dataset'] == train_ds]
            accs = train_subset['Accuracy'].values
            ax.plot(range(len(accs)), accs, marker='o', label=f'Train: {train_ds}', linewidth=2)
        
        ax.set_title(f'{backbone}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Test Dataset Index')
        ax.set_ylim([0.3, 0.85])
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle('Backbone Performance Across Test Datasets', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/performance_trajectory.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/performance_trajectory.png")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def plot_all_results():
    """Generate all graphs."""
    print("\n" + "="*60)
    print("GENERATING PUBLICATION-READY GRAPHS")
    print("="*60 + "\n")
    
    plot_cross_dataset_heatmap()
    plot_backbone_comparison()
    plot_diffusion_comparison()
    plot_f1_heatmap()
    plot_performance_by_dataset()
    generate_results_table()
    
    print("\n" + "="*60)
    print("ALL GRAPHS GENERATED SUCCESSFULLY!")
    print("="*60 + "\n")


if __name__ == '__main__':
    plot_all_results()

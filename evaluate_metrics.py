"""
Comprehensive evaluation metrics for UCF50 video action recognition
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
    cohen_kappa_score, matthews_corrcoef
)
from sklearn.preprocessing import label_binarize
import json
import os
from typing import Dict, List, Tuple
import pandas as pd

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    output_path: str,
    normalize: bool = False
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
    else:
        title = 'Confusion Matrix'
    
    # Create figure with larger size for 50 classes
    plt.figure(figsize=(20, 18))
    
    # Use smaller font for 50 classes
    sns.set(font_scale=0.7)
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Reset font scale
    sns.set(font_scale=1.0)
    
    return cm

def calculate_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str]
) -> pd.DataFrame:
    """Calculate per-class metrics."""
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    support = np.bincount(y_true, minlength=len(classes))
    
    # Create DataFrame
    metrics_df = pd.DataFrame({
        'Class': classes,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    # Sort by F1-Score
    metrics_df = metrics_df.sort_values('F1-Score', ascending=False)
    
    return metrics_df

def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: List[str],
    output_dir: str
):
    """Plot ROC curves for multiclass classification."""
    n_classes = len(classes)
    
    # Binarize the labels
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_bin.ravel(), y_prob.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curves
    plt.figure(figsize=(12, 8))
    
    # Plot micro-average
    plt.plot(
        fpr["micro"], tpr["micro"],
        label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.3f})',
        color='deeppink', linestyle=':', linewidth=4
    )
    
    # Plot ROC for top 10 classes by AUC
    auc_scores = [(i, roc_auc[i]) for i in range(n_classes)]
    auc_scores.sort(key=lambda x: x[1], reverse=True)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    for idx, (class_idx, auc_score) in enumerate(auc_scores[:10]):
        plt.plot(
            fpr[class_idx], tpr[class_idx],
            color=colors[idx], lw=2,
            label=f'{classes[class_idx]} (AUC = {auc_score:.3f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for Top 10 Classes', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return roc_auc

def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: List[str],
    output_dir: str
):
    """Plot precision-recall curves."""
    n_classes = len(classes)
    
    # Binarize the labels
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    # Compute precision-recall curve for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_bin[:, i], y_prob[:, i]
        )
        average_precision[i] = average_precision_score(
            y_true_bin[:, i], y_prob[:, i]
        )
    
    # Compute micro-average
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_prob.ravel()
    )
    average_precision["micro"] = average_precision_score(
        y_true_bin, y_prob, average="micro"
    )
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Plot micro-average
    plt.plot(
        recall["micro"], precision["micro"],
        label=f'Micro-average (AP = {average_precision["micro"]:.3f})',
        color='gold', linestyle=':', linewidth=4
    )
    
    # Plot top 10 classes by AP
    ap_scores = [(i, average_precision[i]) for i in range(n_classes)]
    ap_scores.sort(key=lambda x: x[1], reverse=True)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    for idx, (class_idx, ap_score) in enumerate(ap_scores[:10]):
        plt.plot(
            recall[class_idx], precision[class_idx],
            color=colors[idx], lw=2,
            label=f'{classes[class_idx]} (AP = {ap_score:.3f})'
        )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves for Top 10 Classes', fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    return average_precision

def generate_comprehensive_report(
    results_path: str,
    output_dir: str
):
    """Generate comprehensive evaluation report with all metrics."""
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    y_true = np.array(results['true_labels'])
    y_pred = np.array(results['predictions'])
    y_prob = np.array(results['probabilities'])
    classes = results['categories']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Overall metrics
    overall_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
        'macro_precision': precision_score(y_true, y_pred, average='macro'),
        'macro_recall': recall_score(y_true, y_pred, average='macro'),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred)
    }
    
    # ROC-AUC (multiclass)
    try:
        overall_metrics['roc_auc_ovr'] = roc_auc_score(
            y_true, y_prob, multi_class='ovr', average='macro'
        )
        overall_metrics['roc_auc_ovo'] = roc_auc_score(
            y_true, y_prob, multi_class='ovo', average='macro'
        )
    except:
        print("ROC-AUC calculation failed")
    
    # Plot confusion matrices
    print("Generating confusion matrices...")
    plot_confusion_matrix(
        y_true, y_pred, classes,
        os.path.join(output_dir, 'confusion_matrix.png'),
        normalize=False
    )
    plot_confusion_matrix(
        y_true, y_pred, classes,
        os.path.join(output_dir, 'confusion_matrix_normalized.png'),
        normalize=True
    )
    
    # Per-class metrics
    print("Calculating per-class metrics...")
    per_class_df = calculate_per_class_metrics(y_true, y_pred, classes)
    per_class_df.to_csv(os.path.join(output_dir, 'per_class_metrics.csv'), index=False)
    
    # Plot ROC curves
    print("Generating ROC curves...")
    roc_auc_scores = plot_roc_curves(y_true, y_prob, classes, output_dir)
    
    # Plot Precision-Recall curves
    print("Generating Precision-Recall curves...")
    ap_scores = plot_precision_recall_curves(y_true, y_prob, classes, output_dir)
    
    # Generate text report
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("UCF50 VIDEO ACTION RECOGNITION - EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-"*40 + "\n")
        for metric, value in overall_metrics.items():
            f.write(f"{metric:20s}: {value:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("TOP 10 CLASSES BY F1-SCORE\n")
        f.write("-"*40 + "\n")
        top10 = per_class_df.head(10)
        f.write(top10.to_string(index=False))
        
        f.write("\n\n" + "="*80 + "\n")
        f.write("BOTTOM 10 CLASSES BY F1-SCORE\n")
        f.write("-"*40 + "\n")
        bottom10 = per_class_df.tail(10)
        f.write(bottom10.to_string(index=False))
        
        f.write("\n\n" + "="*80 + "\n")
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("-"*40 + "\n")
        f.write(classification_report(y_true, y_pred, target_names=classes))
    
    print(f"\nEvaluation report saved to {report_path}")
    
    # Save all metrics as JSON
    all_metrics = {
        'overall': overall_metrics,
        'per_class': per_class_df.to_dict('records'),
        'roc_auc_per_class': {classes[i]: float(roc_auc_scores[i]) 
                              for i in range(len(classes))},
        'ap_per_class': {classes[i]: float(ap_scores[i]) 
                         for i in range(len(classes))}
    }
    
    metrics_json_path = os.path.join(output_dir, 'all_metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    return all_metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate evaluation metrics")
    parser.add_argument("--results_path", type=str, required=True,
                       help="Path to test_results.json")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    generate_comprehensive_report(args.results_path, args.output_dir)
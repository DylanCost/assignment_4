"""
Testing script for UCF50 video action recognition
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import json
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from dataset_ucf50 import UCF50Dataset, load_splits
from model_lrcn import LRCN

def test_model(args):
    """Main testing function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data splits
    splits = load_splits(args.splits_path)
    
    # Get test transform
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset
    test_dataset = UCF50Dataset(
        data_dir=args.frame_dir,
        video_list=splits['test']['videos'],
        labels=splits['test']['labels'],
        n_frames=args.n_frames,
        transform=test_transform
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Load model
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    
    # Initialize model
    model = LRCN(
        num_classes=splits['num_classes'],
        cnn_backbone=args.cnn_backbone,
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_num_layers=args.rnn_num_layers,
        dropout=0.0,  # No dropout during testing
        pretrained=False  # We're loading trained weights
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    # Testing
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for frames, labels in tqdm(test_loader, desc="Testing"):
            frames = frames.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(frames)
            probabilities = torch.softmax(outputs, dim=1)
            
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Classification report
    class_report = classification_report(
        all_labels, all_predictions,
        target_names=splits['categories'],
        output_dict=True
    )
    
    # Print results
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Number of test samples: {len(all_labels)}")
    
    # Save results
    results = {
        'test_accuracy': accuracy,
        'predictions': all_predictions.tolist(),
        'true_labels': all_labels.tolist(),
        'probabilities': all_probabilities.tolist(),
        'classification_report': class_report,
        'categories': splits['categories']
    }
    
    results_path = os.path.join(args.output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return results, all_predictions, all_labels, all_probabilities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LRCN on UCF50")
    
    # Data arguments
    parser.add_argument("--frame_dir", type=str, default="data/UCF50_frames",
                       help="Directory containing extracted frames")
    parser.add_argument("--splits_path", type=str, default="data/ucf50_splits.pkl",
                       help="Path to load data splits")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint")
    
    # Model arguments (should match training)
    parser.add_argument("--n_frames", type=int, default=16,
                       help="Number of frames per video")
    parser.add_argument("--cnn_backbone", type=str, default="resnet50",
                       help="CNN backbone architecture")
    parser.add_argument("--rnn_hidden_size", type=int, default=256,
                       help="Hidden size of LSTM")
    parser.add_argument("--rnn_num_layers", type=int, default=2,
                       help="Number of LSTM layers")
    
    # Other arguments
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test model
    results, predictions, labels, probabilities = test_model(args)
"""
Training script for UCF50 video action recognition
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import json
import random
from datetime import datetime
from typing import Tuple
import wandb  # Optional: for experiment tracking

from dataset_ucf50 import UCF50Dataset, create_data_splits, save_splits, load_splits
from model_lrcn import LRCN

def get_data_transforms():
    """Get data transformations for training and validation."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (frames, labels) in enumerate(pbar):
        frames = frames.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(frames)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch_idx, (frames, labels) in enumerate(pbar):
            frames = frames.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def train_model(args):
    """Main training function."""

    # Set ALL random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For DataLoader reproducibility
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create or load data splits
    if os.path.exists(args.splits_path):
        print(f"Loading existing splits from {args.splits_path}")
        splits = load_splits(args.splits_path)
    else:
        print("Creating new data splits...")
        splits = create_data_splits(
            data_dir=args.frame_dir,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            random_state=args.seed
        )
        save_splits(splits, args.splits_path)
    
    # Get transforms
    train_transform, val_transform = get_data_transforms()
    
    # Create datasets
    train_dataset = UCF50Dataset(
        data_dir=args.frame_dir,
        video_list=splits['train']['videos'],
        labels=splits['train']['labels'],
        n_frames=args.n_frames,
        transform=train_transform
    )
    
    val_dataset = UCF50Dataset(
        data_dir=args.frame_dir,
        video_list=splits['val']['videos'],
        labels=splits['val']['labels'],
        n_frames=args.n_frames,
        transform=val_transform
    )
    
    # Create dataloaders with seed control
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        worker_init_fn=seed_worker,  
        generator=g  
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        worker_init_fn=seed_worker,  
        generator=g  
    )
    
    # Initialize model
    model = LRCN(
        num_classes=splits['num_classes'],
        cnn_backbone=args.cnn_backbone,
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_num_layers=args.rnn_num_layers,
        dropout=args.dropout,
        pretrained=args.pretrained
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training log file
    log_file = os.path.join(args.checkpoint_dir, 'training_log.txt')
    
    with open(log_file, 'w') as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write(f"Arguments: {args}\n")
        f.write("="*50 + "\n")
    
    for epoch in range(1, args.n_epochs + 1):
        print(f"\nEpoch {epoch}/{args.n_epochs}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save statistics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Log results
        log_msg = (f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(log_msg)
        
        with open(log_file, 'a') as f:
            f.write(log_msg + "\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'args': args
            }
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best model with val_acc: {val_acc:.2f}%")
        
        # Save regular checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc
            }, checkpoint_path)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }
    
    history_path = os.path.join(args.checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LRCN on UCF50")
    
    # Data arguments
    parser.add_argument("--frame_dir", type=str, default="data/UCF50_frames",
                       help="Directory containing extracted frames")
    parser.add_argument("--splits_path", type=str, default="data/ucf50_splits.pkl",
                       help="Path to save/load data splits")
    parser.add_argument("--n_frames", type=int, default=16,
                       help="Number of frames per video")
    
    # Model arguments
    parser.add_argument("--cnn_backbone", type=str, default="resnet50",
                       choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                       help="CNN backbone architecture")
    parser.add_argument("--rnn_hidden_size", type=int, default=256,
                       help="Hidden size of LSTM")
    parser.add_argument("--rnn_num_layers", type=int, default=2,
                       help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.5,
                       help="Dropout probability")
    parser.add_argument("--pretrained", action="store_true", default=True,
                       help="Use pretrained CNN weights")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    
    # Split arguments
    parser.add_argument("--train_size", type=float, default=0.7,
                       help="Training set proportion")
    parser.add_argument("--val_size", type=float, default=0.15,
                       help="Validation set proportion")
    parser.add_argument("--test_size", type=float, default=0.15,
                       help="Test set proportion")
    
    # Other arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ucf50",
                       help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=5,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Train model
    model, history = train_model(args)
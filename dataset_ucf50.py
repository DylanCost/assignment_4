"""
Dataset module for UCF50 with train/val/test split
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
import pickle
import json
import random

class UCF50Dataset(Dataset):
    """UCF50 Dataset for video action recognition."""
    
    def __init__(
        self,
        data_dir: str,
        video_list: List[str],
        labels: List[int],
        n_frames: int = 16,
        transform=None
    ):
        """
        Args:
            data_dir: Directory containing frame folders
            video_list: List of video names
            labels: List of corresponding labels
            n_frames: Number of frames per video
            transform: Transformations to apply
        """
        self.data_dir = data_dir
        self.video_list = video_list
        self.labels = labels
        self.n_frames = n_frames
        self.transform = transform
        
        # Get action categories
        self.categories = sorted(os.listdir(data_dir))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, idx):
        video_name = self.video_list[idx]
        label = self.labels[idx]
        category = self.categories[label]
        
        # Load frames
        video_dir = os.path.join(self.data_dir, category, video_name)
        frames = []
        
        frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
        
        # Sample frames uniformly if we have more than n_frames
        if len(frame_files) > self.n_frames:
            indices = np.linspace(0, len(frame_files)-1, self.n_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        elif len(frame_files) < self.n_frames:
            # Pad with last frame if we have fewer frames
            frame_files = frame_files + [frame_files[-1]] * (self.n_frames - len(frame_files))
        
        for frame_file in frame_files[:self.n_frames]:
            frame_path = os.path.join(video_dir, frame_file)
            frame = Image.open(frame_path).convert('RGB')
            
            if self.transform:
                frame = self.transform(frame)
            
            frames.append(frame)
        
        # Stack frames: (n_frames, C, H, W)
        frames = torch.stack(frames)
        
        return frames, label

def create_data_splits(
    data_dir: str,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Dict:
    """
    Create train/validation/test splits for UCF50 with group awareness.
    Prevents videos from the same group appearing in different splits.
    """
    import numpy as np
    from collections import defaultdict
    from sklearn.model_selection import train_test_split
    
    np.random.seed(random_state)
    random.seed(random_state)
    
    # Get all categories
    categories = sorted([d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))])
    
    # Group videos by their group ID
    all_videos = []
    all_labels = []
    all_groups = []
    
    for idx, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        videos = [d for d in os.listdir(category_path) 
                 if os.path.isdir(os.path.join(category_path, d))]
        
        for video in videos:
            # Extract group from video name (v_ActionName_g##_c##)
            group_id = None
            if video.startswith('v_'):
                parts = video.split('_')
                for i, part in enumerate(parts):
                    if part.startswith('g') and len(part) > 1 and part[1:].isdigit():
                        # Group ID = category + group number
                        group_id = f"{category}_g{part[1:]}"
                        break
            
            if group_id is None:
                # Fallback: treat each video as its own group
                group_id = f"{category}_{video}"
            
            all_videos.append(video)
            all_labels.append(idx)
            all_groups.append(group_id)
    
    # Convert to numpy arrays
    all_videos = np.array(all_videos)
    all_labels = np.array(all_labels)
    all_groups = np.array(all_groups)
    
    # Get unique groups
    unique_groups = np.unique(all_groups)
    print(f"Found {len(unique_groups)} unique groups across all categories")
    
    # Create group-level labels for stratification
    group_labels = []
    for group in unique_groups:
        group_mask = all_groups == group
        group_label = all_labels[group_mask][0]
        group_labels.append(group_label)
    
    group_labels = np.array(group_labels)
    
    # Split groups (not individual videos)
    groups_temp, groups_test, labels_temp, labels_test = train_test_split(
        unique_groups, group_labels,
        test_size=test_size,
        stratify=group_labels,
        random_state=random_state
    )
    
    val_relative_size = val_size / (train_size + val_size)
    groups_train, groups_val, labels_train, labels_val = train_test_split(
        groups_temp, labels_temp,
        test_size=val_relative_size,
        stratify=labels_temp,
        random_state=random_state
    )
    
    # Assign videos based on their groups
    train_videos, train_labels = [], []
    val_videos, val_labels = [], []
    test_videos, test_labels = [], []
    
    for i, video in enumerate(all_videos):
        group = all_groups[i]
        label = all_labels[i]
        
        if group in groups_train:
            train_videos.append(video)
            train_labels.append(label)
        elif group in groups_val:
            val_videos.append(video)
            val_labels.append(label)
        elif group in groups_test:
            test_videos.append(video)
            test_labels.append(label)
    
    # Create split dictionary
    splits = {
        'train': {'videos': train_videos, 'labels': train_labels},
        'val': {'videos': val_videos, 'labels': val_labels},
        'test': {'videos': test_videos, 'labels': test_labels},
        'categories': categories,
        'num_classes': len(categories)
    }
    
    # Print split statistics
    print(f"Dataset splits created (group-aware):")
    print(f"  Train: {len(train_videos)} videos from {len(groups_train)} groups")
    print(f"  Val: {len(val_videos)} videos from {len(groups_val)} groups")  
    print(f"  Test: {len(test_videos)} videos from {len(groups_test)} groups")
    print(f"  Total: {len(all_videos)} videos")
    print(f"  Classes: {len(categories)}")
    
    return splits

def save_splits(splits: Dict, output_path: str = "data/ucf50_splits.pkl"):
    """Save data splits to file."""
    with open(output_path, 'wb') as f:
        pickle.dump(splits, f)
    print(f"Splits saved to {output_path}")

def load_splits(splits_path: str = "data/ucf50_splits.pkl") -> Dict:
    """Load data splits from file."""
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    return splits
"""
Preprocessing module for UCF50 dataset
Extracts frames from videos using uniform random sampling
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import random
from pathlib import Path
import argparse
from typing import List, Tuple

def get_frames_uniform_sampling(
    video_path: str, 
    n_frames: int = 16,
    resize_to: Tuple[int, int] = (224, 224),
    seed: int = 43
) -> np.ndarray:
    """
    Extract frames from video using uniform random sampling.
    
    Args:
        video_path: Path to video file
        n_frames: Number of frames to extract
        resize_to: Target size for frames
    
    Returns:
        Array of frames with shape (n_frames, H, W, C)
    """
    np.random.seed(seed)
    random.seed(seed)

    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames < n_frames:
        # If video has fewer frames than required, repeat frames
        frame_indices = np.arange(total_frames)
        frame_indices = np.concatenate([
            frame_indices, 
            np.random.choice(frame_indices, n_frames - total_frames)
        ])
    else:
        # Uniform random sampling
        # Divide video into n_frames segments and randomly sample from each
        segment_size = total_frames // n_frames
        frame_indices = []
        for i in range(n_frames):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size if i < n_frames - 1 else total_frames
            frame_indices.append(np.random.randint(start_idx, end_idx))
        frame_indices = np.array(frame_indices)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame
            frame = cv2.resize(frame, resize_to)
            frames.append(frame)
        else:
            # If frame reading fails, use a black frame
            frames.append(np.zeros((*resize_to, 3), dtype=np.uint8))
    
    cap.release()
    return np.array(frames)

def store_frames(
    frames: np.ndarray,
    output_dir: str,
    video_name: str
) -> None:
    """
    Store extracted frames as individual images.
    
    Args:
        frames: Array of frames
        output_dir: Directory to save frames
        video_name: Name of the video (without extension)
    """
    video_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_dir, exist_ok=True)
    
    for i, frame in enumerate(frames):
        frame_path = os.path.join(video_dir, f"frame_{i:04d}.jpg")
        # Convert RGB back to BGR for cv2.imwrite
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_path, frame_bgr)

def preprocess_ucf50(
    data_dir: str = "data/UCF50",
    output_dir: str = "data/UCF50_frames",
    n_frames: int = 16,
    resize_to: Tuple[int, int] = (224, 224),
    seed: int = 42
) -> None:
    """
    Preprocess all UCF50 videos.
    
    Args:
        data_dir: Directory containing UCF50 videos
        output_dir: Directory to save extracted frames
        n_frames: Number of frames to extract per video
        resize_to: Target size for frames
    """

    np.random.seed(seed)
    random.seed(seed)

    # Get all action categories
    action_categories = [d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))]
    action_categories.sort()
    
    print(f"Found {len(action_categories)} action categories")
    
    # Process each category
    for category in tqdm(action_categories, desc="Processing categories"):
        category_path = os.path.join(data_dir, category)
        output_category_path = os.path.join(output_dir, category)
        os.makedirs(output_category_path, exist_ok=True)
        
        # Get all videos in category
        videos = [f for f in os.listdir(category_path) 
                 if f.endswith(('.avi', '.mp4', '.mov'))]
        
        for video in tqdm(videos, desc=f"Processing {category}", leave=False):
            video_path = os.path.join(category_path, video)
            video_name = os.path.splitext(video)[0]
            
            try:
                # Extract frames
                frames = get_frames_uniform_sampling(
                    video_path, n_frames, resize_to
                )
                
                # Store frames
                store_frames(frames, output_category_path, video_name)
                
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess UCF50 dataset")
    parser.add_argument("--data_dir", type=str, default="data/UCF50",
                       help="Path to UCF50 dataset")
    parser.add_argument("--output_dir", type=str, default="data/UCF50_frames",
                       help="Path to save extracted frames")
    parser.add_argument("--n_frames", type=int, default=16,
                       help="Number of frames to extract per video")
    parser.add_argument("--resize", type=int, nargs=2, default=[224, 224],
                       help="Target size for frames")
    
    args = parser.parse_args()
    
    preprocess_ucf50(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_frames=args.n_frames,
        resize_to=tuple(args.resize)
    )
    
    print("Preprocessing complete!")
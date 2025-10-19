#!/bin/bash
python test_ucf50.py \
    --checkpoint_path checkpoints/ucf50/best_model.pth \
    --frame_dir data/UCF50_frames \
    --output_dir results

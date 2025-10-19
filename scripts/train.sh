#!/bin/bash
python train_ucf50.py \
    --frame_dir data/UCF50_frames \
    --cnn_backbone resnet50 \
    --rnn_hidden_size 256 \
    --rnn_num_layers 2 \
    --n_epochs 30 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --pretrained \
    --checkpoint_dir checkpoints/ucf50
    --num_workers 4 \
    --seed 42  
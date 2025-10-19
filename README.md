# UCF50 Video Action Recognition with LRCN

This project implements video action recognition on the UCF50 dataset using a Long-term Recurrent Convolutional Network (LRCN) architecture.

## Requirements

- Python 3.10+
- PyTorch 1.12+
- CUDA 11.3+ (for GPU training)
- 16GB+ RAM
- 50GB+ free disk space

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ucf50-action-recognition.git
cd ucf50-action-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

1. Download the UCF50 dataset:
```bash
bash scripts/download_dataset.sh
```

2. Preprocess videos and extract frames:
```bash
python preprocess_ucf50.py --data_dir data/UCF50 --output_dir data/UCF50_frames --n_frames 16
```

This will extract 16 frames per video using uniform random sampling.

## Training

### Quick Start
```bash
bash scripts/train.sh
```

### Custom Training
```bash
python train_ucf50.py \
    --frame_dir data/UCF50_frames \
    --cnn_backbone resnet50 \
    --n_epochs 30 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --pretrained
```

### Training Arguments
- `--frame_dir`: Directory containing preprocessed frames
- `--cnn_backbone`: CNN architecture (resnet18/34/50/101/152)
- `--rnn_hidden_size`: LSTM hidden dimension (default: 256)
- `--rnn_num_layers`: Number of LSTM layers (default: 2)
- `--n_epochs`: Number of training epochs (default: 30)
- `--batch_size`: Batch size (default: 16)
- `--learning_rate`: Initial learning rate (default: 1e-4)
- `--dropout`: Dropout probability (default: 0.5)
- `--pretrained`: Use ImageNet pretrained weights

## Testing

### Evaluate trained model:
```bash
python test_ucf50.py \
    --checkpoint_path checkpoints/ucf50/best_model.pth \
    --frame_dir data/UCF50_frames
```

### Generate comprehensive metrics:
```bash
python evaluate_metrics.py \
    --results_path results/test_results.json \
    --output_dir evaluation_results
```


## Project Structure
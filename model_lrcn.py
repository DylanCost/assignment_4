"""
LRCN (Long-term Recurrent Convolutional Network) model for UCF50
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

class LRCN(nn.Module):
    """
    LRCN model combining CNN (ResNet) and RNN (LSTM) for video classification.
    """
    
    def __init__(
        self,
        num_classes: int = 50,
        cnn_backbone: str = 'resnet50',
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 2,
        dropout: float = 0.5,
        pretrained: bool = True
    ):
        """
        Args:
            num_classes: Number of action classes
            cnn_backbone: CNN backbone model name
            rnn_hidden_size: Hidden size of LSTM
            rnn_num_layers: Number of LSTM layers
            dropout: Dropout probability
            pretrained: Whether to use pretrained CNN weights
        """
        super(LRCN, self).__init__()
        
        # Initialize CNN backbone
        if cnn_backbone == 'resnet18':
            self.cnn = models.resnet18(pretrained=pretrained)
            cnn_output_size = 512
        elif cnn_backbone == 'resnet34':
            self.cnn = models.resnet34(pretrained=pretrained)
            cnn_output_size = 512
        elif cnn_backbone == 'resnet50':
            self.cnn = models.resnet50(pretrained=pretrained)
            cnn_output_size = 2048
        elif cnn_backbone == 'resnet101':
            self.cnn = models.resnet101(pretrained=pretrained)
            cnn_output_size = 2048
        elif cnn_backbone == 'resnet152':
            self.cnn = models.resnet152(pretrained=pretrained)
            cnn_output_size = 2048
        else:
            raise ValueError(f"Unknown CNN backbone: {cnn_backbone}")
        
        # Remove the final classification layer
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=dropout if rnn_num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Final classification layer
        self.fc = nn.Linear(rnn_hidden_size, num_classes)

        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with fixed seed for reproducibility."""
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
            elif 'fc.weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'fc.bias' in name:
                param.data.fill_(0)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, n_frames, C, H, W)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size, n_frames, c, h, w = x.shape
        
        # Reshape for CNN: (batch_size * n_frames, C, H, W)
        x = x.view(-1, c, h, w)
        
        # Extract CNN features
        with torch.no_grad() if self.training else torch.enable_grad():
            cnn_features = self.cnn(x)  # (batch_size * n_frames, cnn_output_size, 1, 1)
        
        # Flatten CNN output
        cnn_features = cnn_features.view(batch_size, n_frames, -1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(cnn_features)  # (batch_size, n_frames, rnn_hidden_size)
        
        # Use the last timestep output
        last_output = lstm_out[:, -1, :]  # (batch_size, rnn_hidden_size)
        
        # Dropout
        last_output = self.dropout(last_output)
        
        # Classification
        logits = self.fc(last_output)  # (batch_size, num_classes)
        
        return logits
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECG_1D_CNN(nn.Module):
    def __init__(self, input_length=180, num_classes=5, dropout_rate=0.5):
        super(ECG_1D_CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate flattened size
        self.flattened_size = 128 * (input_length // 8)  # After 3 pooling layers
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x, return_features=False):
        # Input shape: (batch_size, 180) -> (batch_size, 1, 180)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        features = F.relu(self.fc2(x))
        x = self.dropout2(features)
        
        logits = self.fc3(x)
        
        if return_features:
            return logits, features
        return logits

    def enable_dropout(self):
        """Enable dropout for uncertainty estimation"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
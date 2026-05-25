import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.cnn_model import ECG_1D_CNN

class ECGTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        for batch_x, batch_y in tqdm(train_loader, desc="Training"):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc="Validation"):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1

def main():
    print("🧠 Starting Single CNN Model Training...")
    
    # Load processed data
    with open('data/processed/datasets.pkl', 'rb') as f:
        datasets = pickle.load(f)
    
    # Prepare data
    X_train, y_train = datasets['train']
    X_val, y_val = datasets['val']
    
    print(f"📊 Training data: {X_train.shape}")
    print(f"📊 Validation data: {X_val.shape}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model and trainer
    model = ECG_1D_CNN(input_length=180, num_classes=5)
    trainer = ECGTrainer(model)
    
    print(f"🎯 Using device: {trainer.device}")
    print(f"📋 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_val_f1 = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(50):
        print(f"\n📈 Epoch {epoch+1}/50")
        
        # Train
        train_loss, train_acc, train_f1 = trainer.train_epoch(train_loader)
        
        # Validate
        val_loss, val_acc, val_f1 = trainer.validate(val_loader)
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            
            # Save best model
            os.makedirs('models/saved_models', exist_ok=True)
            torch.save(model.state_dict(), 'models/saved_models/best_single_cnn.pth')
            print("💾 Saved best model!")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"⏰ Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n🎉 Training completed! Best validation F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    main()
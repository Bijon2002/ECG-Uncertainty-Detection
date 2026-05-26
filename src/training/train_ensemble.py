import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.cnn_model import ECG_1D_CNN
from src.training.train_single import ECGTrainer

class EnsembleTrainer:
    def __init__(self, num_models=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.num_models = num_models
        self.device = device
        self.models = []
        self.trainers = []
        
    def create_ensemble(self):
        """Create multiple CNN models with different initializations"""
        print(f"🔄 Creating ensemble of {self.num_models} models...")
        
        for i in range(self.num_models):
            # Create model with different random seed
            torch.manual_seed(42 + i)
            model = ECG_1D_CNN(input_length=180, num_classes=5)
            trainer = ECGTrainer(model, self.device)
            
            self.models.append(model)
            self.trainers.append(trainer)
            
        print(f"✅ Created {self.num_models} models successfully!")
    
    def train_ensemble(self, train_loader, val_loader, epochs=30):
        """Train all models in the ensemble"""
        ensemble_results = []
        
        for i, trainer in enumerate(self.trainers):
            print(f"\n🧠 Training Model {i+1}/{self.num_models}")
            print("=" * 50)
            
            best_val_f1 = 0
            patience = 8
            patience_counter = 0
            
            model_results = {
                'train_losses': [],
                'val_losses': [],
                'train_f1s': [],
                'val_f1s': []
            }
            
            for epoch in range(epochs):
                # Train
                train_loss, train_acc, train_f1 = trainer.train_epoch(train_loader)
                
                # Validate
                val_loss, val_acc, val_f1 = trainer.validate(val_loader)
                
                # Store results
                model_results['train_losses'].append(train_loss)
                model_results['val_losses'].append(val_loss)
                model_results['train_f1s'].append(train_f1)
                model_results['val_f1s'].append(val_f1)
                
                if (epoch + 1) % 5 == 0:  # Print every 5 epochs
                    print(f"Epoch {epoch+1:2d}: Train F1={train_f1:.4f}, Val F1={val_f1:.4f}")
                
                # Early stopping
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    
                    # Save best model
                    os.makedirs('models/saved_models', exist_ok=True)
                    torch.save(trainer.model.state_dict(), 
                             f'models/saved_models/ensemble_model_{i+1}.pth')
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"⏰ Early stopping at epoch {epoch+1}")
                    break
            
            print(f"✅ Model {i+1} completed! Best Val F1: {best_val_f1:.4f}")
            ensemble_results.append(model_results)
        
        return ensemble_results
    
    def evaluate_ensemble(self, test_loader):
        """Evaluate ensemble performance"""
        print("\n🔍 Evaluating Ensemble Performance...")
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        # Set all models to evaluation mode
        for model in self.models:
            model.eval()
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(test_loader, desc="Testing"):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Get predictions from all models
                batch_predictions = []
                batch_probabilities = []
                
                for model in self.models:
                    outputs = model(batch_x)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    
                    batch_predictions.append(preds.cpu().numpy())
                    batch_probabilities.append(probs.cpu().numpy())
                
                # Ensemble predictions (majority vote)
                batch_predictions = np.array(batch_predictions)  # (num_models, batch_size)
                ensemble_preds = []
                
                for i in range(batch_predictions.shape[1]):
                    # Majority vote for each sample
                    votes = batch_predictions[:, i]
                    ensemble_pred = np.bincount(votes).argmax()
                    ensemble_preds.append(ensemble_pred)
                
                # Average probabilities
                batch_probabilities = np.array(batch_probabilities)  # (num_models, batch_size, num_classes)
                avg_probs = np.mean(batch_probabilities, axis=0)
                
                all_predictions.extend(ensemble_preds)
                all_probabilities.append(avg_probs)
                all_labels.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print(f"📊 Ensemble Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        return {
            'predictions': all_predictions,
            'probabilities': np.vstack(all_probabilities),
            'labels': all_labels,
            'accuracy': accuracy,
            'f1_score': f1
        }

def main():
    print("🚀 Starting Deep Ensemble Training...")
    
    # Load processed data
    with open('data/processed/datasets.pkl', 'rb') as f:
        datasets = pickle.load(f)
    
    # Prepare data
    X_train, y_train = datasets['train']
    X_val, y_val = datasets['val']
    X_test, y_test = datasets['test']
    
    print(f"📊 Training data: {X_train.shape}")
    print(f"📊 Validation data: {X_val.shape}")
    print(f"📊 Test data: {X_test.shape}")
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.LongTensor(y_test)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create and train ensemble
    ensemble_trainer = EnsembleTrainer(num_models=5)
    ensemble_trainer.create_ensemble()
    
    print(f"🎯 Using device: {ensemble_trainer.device}")
    
    # Train ensemble
    results = ensemble_trainer.train_ensemble(train_loader, val_loader, epochs=30)
    
    # Evaluate ensemble
    test_results = ensemble_trainer.evaluate_ensemble(test_loader)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/ensemble_results.pkl', 'wb') as f:
        pickle.dump({
            'training_results': results,
            'test_results': test_results
        }, f)
    
    print(f"\n🎉 Ensemble Training Complete!")
    print(f"📁 Results saved to: results/ensemble_results.pkl")

if __name__ == "__main__":
    main()
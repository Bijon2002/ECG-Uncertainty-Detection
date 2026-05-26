import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import pickle
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.cnn_model import ECG_1D_CNN

class UncertaintyQuantifier:
    def __init__(self, models, device='cpu'):
        self.models = models
        self.device = device
        self.class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
        
    def monte_carlo_dropout(self, X, num_samples=100):
        """Monte Carlo Dropout for epistemic uncertainty"""
        print("🔄 Computing Monte Carlo Dropout uncertainty...")
        
        all_predictions = []
        
        for model in self.models:
            model.eval()
            # Enable dropout during inference
            model.enable_dropout()
            
            model_predictions = []
            
            with torch.no_grad():
                for _ in range(num_samples):
                    X_tensor = torch.FloatTensor(X).to(self.device)
                    if X_tensor.dim() == 2:
                        X_tensor = X_tensor.unsqueeze(1)
                    
                    outputs = model(X_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    model_predictions.append(probs.cpu().numpy())
            
            all_predictions.append(np.array(model_predictions))
        
        # Combine predictions from all models
        ensemble_predictions = np.concatenate(all_predictions, axis=0)  # (total_samples, n_data, n_classes)
        
        # Calculate mean and variance
        mean_probs = np.mean(ensemble_predictions, axis=0)
        variance = np.var(ensemble_predictions, axis=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = np.mean(variance, axis=1)
        
        return {
            'predictions': mean_probs,
            'epistemic_uncertainty': epistemic_uncertainty,
            'all_samples': ensemble_predictions
        }
    
    def predictive_entropy(self, X):
        """Predictive entropy for overall confidence"""
        print("🔄 Computing Predictive Entropy...")
        
        # Get ensemble predictions
        all_probs = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                if X_tensor.dim() == 2:
                    X_tensor = X_tensor.unsqueeze(1)
                
                outputs = model(X_tensor)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        # Average predictions
        mean_probs = np.mean(all_probs, axis=0)
        
        # Calculate entropy
        epsilon = 1e-8  # Small value to avoid log(0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=1)
        
        return {
            'predictions': mean_probs,
            'entropy': entropy,
            'max_entropy': np.log(len(self.class_names))  # Maximum possible entropy
        }
    
    def cluster_based_entropy(self, X, n_clusters=5):
        """Cluster-based entropy for medical category-specific uncertainty"""
        print("🔄 Computing Cluster-based Entropy...")
        
        # Get ensemble predictions
        all_probs = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                if X_tensor.dim() == 2:
                    X_tensor = X_tensor.unsqueeze(1)
                
                outputs = model(X_tensor)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        # Average predictions
        mean_probs = np.mean(all_probs, axis=0)
        predicted_classes = np.argmax(mean_probs, axis=1)

        # KMeans cannot fit more clusters than available samples.
        # For single-sample inference, fall back to entropy-based uncertainty.
        n_samples = mean_probs.shape[0]
        effective_clusters = min(n_clusters, n_samples)
        epsilon = 1e-8
        base_entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=1)

        if effective_clusters < 2:
            max_entropy = np.log(mean_probs.shape[1])
            normalized_entropy = base_entropy / max_entropy if max_entropy > 0 else np.zeros_like(base_entropy)

            return {
                'predictions': mean_probs,
                'cluster_uncertainty': normalized_entropy,
                'clusters': np.zeros(n_samples, dtype=int),
                'cluster_centers': mean_probs.copy()
            }
        
        # Cluster samples based on prediction probabilities
        kmeans = KMeans(n_clusters=effective_clusters, random_state=42)
        clusters = kmeans.fit_predict(mean_probs)
        
        # Calculate uncertainty within each cluster
        cluster_uncertainties = np.zeros(n_samples)
        
        for cluster_id in range(effective_clusters):
            cluster_mask = clusters == cluster_id
            if np.sum(cluster_mask) > 0:
                cluster_probs = mean_probs[cluster_mask]
                
                # Calculate entropy within cluster
                cluster_entropy = -np.sum(cluster_probs * np.log(cluster_probs + epsilon), axis=1)
                
                # Normalize by cluster-specific maximum entropy
                cluster_classes = np.unique(predicted_classes[cluster_mask])
                max_cluster_entropy = np.log(len(cluster_classes)) if len(cluster_classes) > 1 else 0
                
                if max_cluster_entropy > 0:
                    normalized_entropy = cluster_entropy / max_cluster_entropy
                else:
                    normalized_entropy = np.zeros_like(cluster_entropy)
                
                cluster_uncertainties[cluster_mask] = normalized_entropy
        
        return {
            'predictions': mean_probs,
            'cluster_uncertainty': cluster_uncertainties,
            'clusters': clusters,
            'cluster_centers': kmeans.cluster_centers_
        }

def main():
    print("🚀 Starting Uncertainty Quantification...")
    
    # Load processed data
    with open('data/processed/datasets.pkl', 'rb') as f:
        datasets = pickle.load(f)
    
    # Load trained ensemble models
    models = []
    for i in range(1, 6):
        model = ECG_1D_CNN(input_length=180, num_classes=5)
        model.load_state_dict(torch.load(f'models/saved_models/ensemble_model_{i}.pth'))
        model.eval()
        models.append(model)
    
    print(f"✅ Loaded {len(models)} ensemble models")
    
    # Prepare test data (smaller subset for demonstration)
    X_test, y_test = datasets['test']
    X_val, y_val = datasets['val']
    
    # Use subset for faster computation
    np.random.seed(42)  # For reproducibility
    test_indices = np.random.choice(len(X_test), size=1000, replace=False)
    X_test_subset = X_test[test_indices]
    y_test_subset = y_test[test_indices]
    
    print(f"📊 Testing on {len(X_test_subset)} samples")
    
    # Initialize uncertainty quantifier
    uncertainty_quantifier = UncertaintyQuantifier(models)
    
    # 1. Monte Carlo Dropout
    mc_results = uncertainty_quantifier.monte_carlo_dropout(X_test_subset, num_samples=50)
    
    # 2. Predictive Entropy
    entropy_results = uncertainty_quantifier.predictive_entropy(X_test_subset)
    
    # 3. Cluster-based Entropy
    cluster_results = uncertainty_quantifier.cluster_based_entropy(X_test_subset)
    
    # Combine results
    uncertainty_results = {
        'monte_carlo': mc_results,
        'predictive_entropy': entropy_results,
        'cluster_based': cluster_results,
        'true_labels': y_test_subset,
        'test_indices': test_indices
    }
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/uncertainty_results.pkl', 'wb') as f:
        pickle.dump(uncertainty_results, f)
    
    print("🎉 Uncertainty Quantification Complete!")
    print("📁 Results saved to: results/uncertainty_results.pkl")
    
    # Print summary statistics
    print("\n📊 Uncertainty Summary:")
    print(f"Monte Carlo - Mean Epistemic Uncertainty: {np.mean(mc_results['epistemic_uncertainty']):.4f}")
    print(f"Predictive Entropy - Mean Entropy: {np.mean(entropy_results['entropy']):.4f}")
    print(f"Cluster-based - Mean Uncertainty: {np.mean(cluster_results['cluster_uncertainty']):.4f}")

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.calibration import calibration_curve
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

class UncertaintyEvaluator:
    def __init__(self, results_path='results/uncertainty_results.pkl'):
        with open(results_path, 'rb') as f:
            self.results = pickle.load(f)
        
        self.class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
        
    def evaluate_predictions(self):
        """Evaluate prediction accuracy for each method"""
        print("📊 Evaluating Prediction Accuracy...")
        
        true_labels = self.results['true_labels']
        
        methods = {
            'Monte Carlo': self.results['monte_carlo']['predictions'],
            'Predictive Entropy': self.results['predictive_entropy']['predictions'],
            'Cluster-based': self.results['cluster_based']['predictions']
        }
        
        results = {}
        
        for method_name, predictions in methods.items():
            pred_labels = np.argmax(predictions, axis=1)
            
            accuracy = accuracy_score(true_labels, pred_labels)
            f1 = f1_score(true_labels, pred_labels, average='weighted')
            
            results[method_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': pred_labels
            }
            
            print(f"{method_name:15} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
    
    def plot_uncertainty_distributions(self):
        """Plot uncertainty distributions for each method"""
        print("📈 Creating uncertainty distribution plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Monte Carlo Dropout
        mc_uncertainty = self.results['monte_carlo']['epistemic_uncertainty']
        axes[0].hist(mc_uncertainty, bins=30, alpha=0.7, color='blue')
        axes[0].set_title('Monte Carlo Dropout\nEpistemic Uncertainty')
        axes[0].set_xlabel('Uncertainty')
        axes[0].set_ylabel('Frequency')
        
        # Predictive Entropy
        entropy = self.results['predictive_entropy']['entropy']
        axes[1].hist(entropy, bins=30, alpha=0.7, color='green')
        axes[1].set_title('Predictive Entropy\nOverall Confidence')
        axes[1].set_xlabel('Entropy')
        axes[1].set_ylabel('Frequency')
        
        # Cluster-based Entropy
        cluster_uncertainty = self.results['cluster_based']['cluster_uncertainty']
        axes[2].hist(cluster_uncertainty, bins=30, alpha=0.7, color='red')
        axes[2].set_title('Cluster-based Entropy\nMedical-Specific Uncertainty')
        axes[2].set_xlabel('Normalized Uncertainty')
        axes[2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('results/uncertainty_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("💾 Saved: results/uncertainty_distributions.png")
    
    def analyze_uncertainty_vs_accuracy(self):
        """Analyze relationship between uncertainty and prediction accuracy"""
        print("🔍 Analyzing uncertainty vs accuracy relationship...")
        
        true_labels = self.results['true_labels']
        
        # Get predictions and uncertainties
        mc_preds = np.argmax(self.results['monte_carlo']['predictions'], axis=1)
        entropy_preds = np.argmax(self.results['predictive_entropy']['predictions'], axis=1)
        cluster_preds = np.argmax(self.results['cluster_based']['predictions'], axis=1)
        
        mc_uncertainty = self.results['monte_carlo']['epistemic_uncertainty']
        entropy_uncertainty = self.results['predictive_entropy']['entropy']
        cluster_uncertainty = self.results['cluster_based']['cluster_uncertainty']
        
        # Calculate correctness
        mc_correct = (mc_preds == true_labels).astype(int)
        entropy_correct = (entropy_preds == true_labels).astype(int)
        cluster_correct = (cluster_preds == true_labels).astype(int)
        
        # Create correlation plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Monte Carlo
        axes[0].scatter(mc_uncertainty, mc_correct, alpha=0.6, s=10)
        axes[0].set_xlabel('Monte Carlo Uncertainty')
        axes[0].set_ylabel('Correct Prediction (1=Yes, 0=No)')
        axes[0].set_title('MC Dropout: Uncertainty vs Correctness')
        
        # Predictive Entropy
        axes[1].scatter(entropy_uncertainty, entropy_correct, alpha=0.6, s=10)
        axes[1].set_xlabel('Predictive Entropy')
        axes[1].set_ylabel('Correct Prediction (1=Yes, 0=No)')
        axes[1].set_title('Entropy: Uncertainty vs Correctness')
        
        # Cluster-based
        axes[2].scatter(cluster_uncertainty, cluster_correct, alpha=0.6, s=10)
        axes[2].set_xlabel('Cluster-based Uncertainty')
        axes[2].set_ylabel('Correct Prediction (1=Yes, 0=No)')
        axes[2].set_title('Cluster-based: Uncertainty vs Correctness')
        
        plt.tight_layout()
        plt.savefig('results/uncertainty_vs_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("💾 Saved: results/uncertainty_vs_accuracy.png")
    
    def create_confusion_matrices(self):
        """Create confusion matrices for each method"""
        print("📊 Creating confusion matrices...")
        
        from sklearn.metrics import confusion_matrix
        
        true_labels = self.results['true_labels']
        
        methods = {
            'Monte Carlo': np.argmax(self.results['monte_carlo']['predictions'], axis=1),
            'Predictive Entropy': np.argmax(self.results['predictive_entropy']['predictions'], axis=1),
            'Cluster-based': np.argmax(self.results['cluster_based']['predictions'], axis=1)
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (method_name, predictions) in enumerate(methods.items()):
            cm = confusion_matrix(true_labels, predictions)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names,
                       ax=axes[idx])
            
            axes[idx].set_title(f'{method_name}\nConfusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("💾 Saved: results/confusion_matrices.png")
    
    def uncertainty_based_rejection(self):
        """Analyze performance with uncertainty-based rejection"""
        print("🎯 Analyzing uncertainty-based rejection...")
        
        true_labels = self.results['true_labels']
        
        methods = {
            'Monte Carlo': {
                'predictions': np.argmax(self.results['monte_carlo']['predictions'], axis=1),
                'uncertainty': self.results['monte_carlo']['epistemic_uncertainty']
            },
            'Predictive Entropy': {
                'predictions': np.argmax(self.results['predictive_entropy']['predictions'], axis=1),
                'uncertainty': self.results['predictive_entropy']['entropy']
            },
            'Cluster-based': {
                'predictions': np.argmax(self.results['cluster_based']['predictions'], axis=1),
                'uncertainty': self.results['cluster_based']['cluster_uncertainty']
            }
        }
        
        rejection_rates = np.arange(0, 0.5, 0.05)  # 0% to 50% rejection
        
        plt.figure(figsize=(12, 8))
        
        for method_name, method_data in methods.items():
            accuracies = []
            
            for rejection_rate in rejection_rates:
                # Sort by uncertainty (descending)
                uncertainty_order = np.argsort(method_data['uncertainty'])[::-1]
                
                # Reject top uncertain samples
                n_reject = int(len(true_labels) * rejection_rate)
                keep_indices = uncertainty_order[n_reject:]
                
                if len(keep_indices) > 0:
                    kept_predictions = method_data['predictions'][keep_indices]
                    kept_labels = true_labels[keep_indices]
                    
                    accuracy = accuracy_score(kept_labels, kept_predictions)
                    accuracies.append(accuracy)
                else:
                    accuracies.append(0)
            
            plt.plot(rejection_rates * 100, accuracies, 
                    marker='o', label=method_name, linewidth=2)
        
        plt.xlabel('Rejection Rate (%)')
        plt.ylabel('Accuracy on Remaining Samples')
        plt.title('Accuracy vs Uncertainty-based Rejection Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results/rejection_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("💾 Saved: results/rejection_analysis.png")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("📋 Generating Summary Report...")
        
        # Evaluate predictions
        pred_results = self.evaluate_predictions()
        
        # Create summary
        summary = {
            'Dataset Size': len(self.results['true_labels']),
            'Class Distribution': np.bincount(self.results['true_labels']),
            'Method Performance': pred_results
        }
        
        print("\n" + "="*50)
        print("📊 UNCERTAINTY QUANTIFICATION SUMMARY REPORT")
        print("="*50)
        
        print(f"\n📈 Dataset Information:")
        print(f"   Test samples: {summary['Dataset Size']}")
        print(f"   Class distribution: {dict(zip(self.class_names, summary['Class Distribution']))}")
        
        print(f"\n🎯 Method Performance:")
        for method, metrics in summary['Method Performance'].items():
            print(f"   {method:15} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        print(f"\n🔍 Uncertainty Statistics:")
        mc_uncertainty = self.results['monte_carlo']['epistemic_uncertainty']
        entropy_uncertainty = self.results['predictive_entropy']['entropy']
        cluster_uncertainty = self.results['cluster_based']['cluster_uncertainty']
        
        print(f"   Monte Carlo     - Mean: {np.mean(mc_uncertainty):.4f}, Std: {np.std(mc_uncertainty):.4f}")
        print(f"   Predictive Entropy - Mean: {np.mean(entropy_uncertainty):.4f}, Std: {np.std(entropy_uncertainty):.4f}")
        print(f"   Cluster-based   - Mean: {np.mean(cluster_uncertainty):.4f}, Std: {np.std(cluster_uncertainty):.4f}")
        
        return summary

def main():
    print("🚀 Starting Comprehensive Uncertainty Evaluation...")
    
    # Initialize evaluator
    evaluator = UncertaintyEvaluator()
    
    # Run all evaluations
    evaluator.evaluate_predictions()
    evaluator.plot_uncertainty_distributions()
    evaluator.analyze_uncertainty_vs_accuracy()
    evaluator.create_confusion_matrices()
    evaluator.uncertainty_based_rejection()
    evaluator.generate_summary_report()
    
    print("\n🎉 Evaluation Complete!")
    print("📁 All plots saved to results/ directory")

if __name__ == "__main__":
    main()
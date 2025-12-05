"""
Model evaluation module for heart disease prediction project.
Provides comprehensive evaluation metrics and visualization tools.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import joblib
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Class to evaluate and compare multiple machine learning models."""
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        self.models = {}
        self.evaluation_results = {}
        
    def load_models(self) -> None:
        """Load all trained models from the models directory."""
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_model.joblib')]
        
        for model_file in model_files:
            model_name = model_file.replace('_model.joblib', '')
            model_path = os.path.join(self.models_dir, model_file)
            self.models[model_name] = joblib.load(model_path)
            print(f"Loaded {model_name} model")
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """
        Evaluate all loaded models on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation results for all models
        """
        if not self.models:
            self.load_models()
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = model.score(X_test, y_test)
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            self.evaluation_results[model_name] = {
                'accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'roc_auc': roc_auc,
                'fpr': fpr,
                'tpr': tpr,
                'precision': precision,
                'recall': recall,
                'avg_precision': avg_precision,
                'predictions': y_pred,
                'prediction_probabilities': y_pred_proba
            }
        
        return self.evaluation_results
    
    def create_comparison_report(self) -> pd.DataFrame:
        """Create a comparison report of all models."""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_all_models first.")
        
        comparison_data = []
        
        for model_name, results in self.evaluation_results.items():
            class_report = results['classification_report']
            
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'ROC-AUC': results['roc_auc'],
                'Precision': class_report['1']['precision'],
                'Recall': class_report['1']['recall'],
                'F1-Score': class_report['1']['f1-score'],
                'Avg Precision': results['avg_precision']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
        
        return comparison_df
    
    def plot_roc_curves(self, figsize: Tuple[int, int] = (10, 8)) -> None:
        """Plot ROC curves for all models."""
        plt.figure(figsize=figsize)
        
        for model_name, results in self.evaluation_results.items():
            plt.plot(results['fpr'], results['tpr'], 
                    label=f"{model_name} (AUC = {results['roc_auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, figsize: Tuple[int, int] = (10, 8)) -> None:
        """Plot Precision-Recall curves for all models."""
        plt.figure(figsize=figsize)
        
        for model_name, results in self.evaluation_results.items():
            plt.plot(results['recall'], results['precision'], 
                    label=f"{model_name} (AP = {results['avg_precision']:.3f})")
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig('results/precision_recall_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot confusion matrices for all models."""
        n_models = len(self.evaluation_results)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, results) in enumerate(self.evaluation_results.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                       cmap='Blues', ax=ax)
            ax.set_title(f'{model_name}\nAccuracy: {results["accuracy"]:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_models, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig('results/confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_metrics_comparison(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot bar chart comparing key metrics across models."""
        comparison_df = self.create_comparison_report()
        
        metrics = ['Accuracy', 'ROC-AUC', 'Precision', 'Recall', 'F1-Score']
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric])
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        # Hide the last subplot
        axes[-1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('results/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_detailed_report(self, save_path: str = 'results/evaluation_report.txt') -> None:
        """Generate a detailed text report of all model evaluations."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write("HEART DISEASE PREDICTION MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary table
            comparison_df = self.create_comparison_report()
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
            
            # Detailed results for each model
            for model_name, results in self.evaluation_results.items():
                f.write(f"DETAILED RESULTS FOR {model_name.upper()}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Accuracy: {results['accuracy']:.4f}\n")
                f.write(f"ROC-AUC: {results['roc_auc']:.4f}\n")
                f.write(f"Average Precision: {results['avg_precision']:.4f}\n\n")
                
                f.write("Classification Report:\n")
                class_report = results['classification_report']
                f.write(f"Class 0 (No Disease): Precision={class_report['0']['precision']:.3f}, "
                       f"Recall={class_report['0']['recall']:.3f}, "
                       f"F1={class_report['0']['f1-score']:.3f}\n")
                f.write(f"Class 1 (Disease): Precision={class_report['1']['precision']:.3f}, "
                       f"Recall={class_report['1']['recall']:.3f}, "
                       f"F1={class_report['1']['f1-score']:.3f}\n")
                
                f.write(f"\nConfusion Matrix:\n")
                conf_matrix = results['confusion_matrix']
                f.write(f"True Negatives: {conf_matrix[0,0]}, False Positives: {conf_matrix[0,1]}\n")
                f.write(f"False Negatives: {conf_matrix[1,0]}, True Positives: {conf_matrix[1,1]}\n")
                f.write("\n" + "="*60 + "\n\n")
        
        print(f"Detailed evaluation report saved to {save_path}")

def main():
    """Main function to run model evaluation."""
    from data_loader import HeartDiseaseDataLoader
    from data_preprocessing import prepare_train_test_data
    
    # Load and prepare data
    loader = HeartDiseaseDataLoader()
    df = loader.load_data()
    X_train, X_test, y_train, y_test = prepare_train_test_data(df)
    
    # Evaluate models
    evaluator = ModelEvaluator()
    evaluator.evaluate_all_models(X_test, y_test)
    
    # Generate comparison report
    comparison_df = evaluator.create_comparison_report()
    print("\nModel Performance Comparison:")
    print(comparison_df)
    
    # Create visualizations
    evaluator.plot_roc_curves()
    evaluator.plot_precision_recall_curves()
    evaluator.plot_confusion_matrices()
    evaluator.plot_metrics_comparison()
    
    # Generate detailed report
    evaluator.generate_detailed_report()
    
    print("\nModel evaluation completed successfully!")

if __name__ == "__main__":
    main()

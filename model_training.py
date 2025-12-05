"""
Model training module for heart disease prediction project.
Implements multiple machine learning classifiers and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class HeartDiseaseModelTrainer:
    """Class to handle training of multiple machine learning models."""
    
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.model_scores = {}
        
        # Define models with their parameter grids
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'knn': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            },
            'naive_bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                }
            }
        }
    
    def train_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                          cv_folds: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Train a single model with hyperparameter tuning.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for grid search
            
        Returns:
            Dictionary containing model results
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not supported")
        
        config = self.model_configs[model_name]
        model = config['model']
        param_grid = config['params']
        
        print(f"Training {model_name}...")
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grid, cv=cv_folds, scoring=scoring,
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store results
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
        
        # Perform cross-validation on the best model
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring=scoring)
        
        results = {
            'model': best_model,
            'best_score': best_score,
            'best_params': best_params,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        self.best_models[model_name] = best_model
        self.model_scores[model_name] = results
        
        print(f"{model_name} - Best CV Score: {best_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Best Parameters: {best_params}")
        
        return results
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        cv_folds: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Train all models with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for grid search
            
        Returns:
            Dictionary containing all model results
        """
        print("Starting model training with hyperparameter tuning...")
        print("=" * 60)
        
        all_results = {}
        
        for model_name in self.model_configs.keys():
            try:
                results = self.train_single_model(model_name, X_train, y_train, cv_folds, scoring)
                all_results[model_name] = results
                print("-" * 40)
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        # Find the best model
        best_model_name = max(all_results.keys(), 
                             key=lambda x: all_results[x]['best_score'])
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Score: {all_results[best_model_name]['best_score']:.4f}")
        
        return all_results
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """
        Evaluate all trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation results for all models
        """
        if not self.best_models:
            raise ValueError("No models have been trained yet")
        
        evaluation_results = {}
        
        print("Evaluating models on test data...")
        print("=" * 50)
        
        for model_name, model in self.best_models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = model.score(X_test, y_test)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            evaluation_results[model_name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'predictions': y_pred,
                'prediction_probabilities': y_pred_proba
            }
            
            print(f"{model_name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")
            print(f"  Precision: {class_report['1']['precision']:.4f}")
            print(f"  Recall: {class_report['1']['recall']:.4f}")
            print(f"  F1-Score: {class_report['1']['f1-score']:.4f}")
            print("-" * 30)
        
        return evaluation_results
    
    def save_models(self, save_dir: str = 'models') -> None:
        """Save all trained models."""
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.best_models.items():
            model_path = os.path.join(save_dir, f'{model_name}_model.joblib')
            joblib.dump(model, model_path)
            print(f"Saved {model_name} to {model_path}")
        
        # Save model scores
        scores_path = os.path.join(save_dir, 'model_scores.joblib')
        joblib.dump(self.model_scores, scores_path)
        print(f"Saved model scores to {scores_path}")
    
    def load_model(self, model_name: str, model_path: str) -> None:
        """Load a trained model."""
        model = joblib.load(model_path)
        self.best_models[model_name] = model
        print(f"Loaded {model_name} from {model_path}")
    
    def get_feature_importance(self, model_name: str, feature_names: list = None) -> pd.DataFrame:
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.best_models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.best_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importance))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            print(f"Model {model_name} does not have feature importance")
            return None

def main():
    """Main function to test model training."""
    from data_loader import HeartDiseaseDataLoader
    from data_preprocessing import prepare_train_test_data
    
    # Load and prepare data
    loader = HeartDiseaseDataLoader()
    df = loader.load_data()
    X_train, X_test, y_train, y_test = prepare_train_test_data(df)
    
    # Train models
    trainer = HeartDiseaseModelTrainer()
    results = trainer.train_all_models(X_train, y_train)
    
    # Evaluate models
    evaluation = trainer.evaluate_models(X_test, y_test)
    
    # Save models
    trainer.save_models()
    
    print("Model training completed successfully!")

if __name__ == "__main__":
    main()

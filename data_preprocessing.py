"""
Data preprocessing module for heart disease prediction project.
Handles data cleaning, feature engineering, and preparation for machine learning.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, Any
import joblib
import os

class HeartDiseasePreprocessor:
    """Class to handle preprocessing of heart disease data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        self.is_fitted = False
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        # For 'ca' and 'thal' columns, use median imputation
        numeric_cols_with_missing = ['ca', 'thal']
        
        for col in numeric_cols_with_missing:
            if col in df_clean.columns and df_clean[col].isnull().any():
                median_value = df_clean[col].median()
                df_clean.loc[:, col] = df_clean[col].fillna(median_value)
                print(f"Filled {col} missing values with median: {median_value}")
        
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df_engineered = df.copy()
        
        # Age groups
        df_engineered['age_group'] = pd.cut(df_engineered['age'], 
                                          bins=[0, 40, 50, 60, 100], 
                                          labels=['young', 'middle_aged', 'senior', 'elderly'])
        
        # BMI approximation (using cholesterol and age as proxy)
        # This is a simplified approach - in real scenarios, you'd have height/weight
        df_engineered['health_score'] = (
            df_engineered['thalach'] / df_engineered['age'] * 
            (1 / (df_engineered['chol'] / 200))
        )
        
        # Blood pressure categories
        df_engineered['bp_category'] = pd.cut(df_engineered['trestbps'],
                                            bins=[0, 120, 140, 180, 300],
                                            labels=['normal', 'elevated', 'high', 'very_high'])
        
        # Cholesterol categories
        df_engineered['chol_category'] = pd.cut(df_engineered['chol'],
                                              bins=[0, 200, 240, 500],
                                              labels=['normal', 'borderline', 'high'])
        
        # Risk factors combination
        df_engineered['risk_factors'] = (
            df_engineered['fbs'] + 
            df_engineered['exang'] + 
            (df_engineered['cp'] == 4).astype(int)  # Asymptomatic chest pain
        )
        
        return df_engineered
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the encoders (True for training, False for prediction)
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        # Categorical columns to encode
        categorical_cols = ['age_group', 'bp_category', 'chol_category']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        unique_values = df_encoded[col].astype(str).unique()
                        known_values = self.label_encoders[col].classes_
                        
                        # Replace unseen categories with the most frequent one
                        for val in unique_values:
                            if val not in known_values:
                                most_frequent = df_encoded[col].mode()[0] if not df_encoded[col].empty else known_values[0]
                                df_encoded[col] = df_encoded[col].replace(val, most_frequent)
                        
                        df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Scale numerical features.
        
        Args:
            X: Feature DataFrame
            fit: Whether to fit the scaler (True for training, False for prediction)
            
        Returns:
            Scaled feature array
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = X.columns.tolist()
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def preprocess(self, df: pd.DataFrame, target_col: str = 'target', 
                   fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            fit: Whether to fit preprocessors (True for training, False for prediction)
            
        Returns:
            Tuple of (X_processed, y) or just X_processed if no target
        """
        # Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # Engineer features
        df_engineered = self.engineer_features(df_clean)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_engineered, fit=fit)
        
        # Separate features and target
        if target_col in df_encoded.columns:
            X = df_encoded.drop(columns=[target_col])
            y = df_encoded[target_col].values
        else:
            X = df_encoded
            y = None
        
        # Scale features
        X_scaled = self.scale_features(X, fit=fit)
        
        if fit:
            self.is_fitted = True
        
        if y is not None:
            return X_scaled, y
        else:
            return X_scaled
    
    def save_preprocessor(self, filepath: str) -> None:
        """Save the fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before saving.")
        
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str) -> None:
        """Load a fitted preprocessor."""
        preprocessor_data = joblib.load(filepath)
        
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.imputer = preprocessor_data['imputer']
        self.feature_names = preprocessor_data['feature_names']
        self.is_fitted = preprocessor_data['is_fitted']
        
        print(f"Preprocessor loaded from {filepath}")

def prepare_train_test_data(df: pd.DataFrame, test_size: float = 0.2, 
                           random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training and testing data.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    preprocessor = HeartDiseasePreprocessor()
    
    # Preprocess the data
    X, y = preprocessor.preprocess(df, fit=True)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Save the preprocessor
    os.makedirs('models', exist_ok=True)
    preprocessor.save_preprocessor('models/preprocessor.joblib')
    
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    print(f"Feature names: {preprocessor.feature_names}")
    
    return X_train, X_test, y_train, y_test

def main():
    """Main function to test preprocessing."""
    from data_loader import HeartDiseaseDataLoader
    
    # Load data
    loader = HeartDiseaseDataLoader()
    df = loader.load_data()
    
    # Prepare train/test data
    X_train, X_test, y_train, y_test = prepare_train_test_data(df)
    
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()

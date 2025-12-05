"""
Data loader module for heart disease prediction project.
Downloads and loads the UCI Heart Disease dataset.
"""

import pandas as pd
import numpy as np
import os
from urllib.request import urlretrieve
from typing import Tuple, Optional

class HeartDiseaseDataLoader:
    """Class to handle loading and basic preprocessing of heart disease data."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        self.dataset_path = os.path.join(data_dir, "heart_disease.csv")
        
        # Column names for the dataset
        self.column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        
        # Feature descriptions
        self.feature_descriptions = {
            'age': 'Age in years',
            'sex': 'Sex (1 = male; 0 = female)',
            'cp': 'Chest pain type (0-3)',
            'trestbps': 'Resting blood pressure (mm Hg)',
            'chol': 'Serum cholesterol (mg/dl)',
            'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
            'restecg': 'Resting electrocardiographic results (0-2)',
            'thalach': 'Maximum heart rate achieved',
            'exang': 'Exercise induced angina (1 = yes; 0 = no)',
            'oldpeak': 'ST depression induced by exercise relative to rest',
            'slope': 'Slope of the peak exercise ST segment (0-2)',
            'ca': 'Number of major vessels (0-3) colored by fluoroscopy',
            'thal': 'Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)',
            'target': 'Heart disease presence (0 = no disease; 1-4 = disease)'
        }
    
    def download_dataset(self) -> None:
        """Download the heart disease dataset if it doesn't exist."""
        if not os.path.exists(self.dataset_path):
            print(f"Downloading dataset to {self.dataset_path}...")
            os.makedirs(self.data_dir, exist_ok=True)
            urlretrieve(self.dataset_url, self.dataset_path)
            print("Dataset downloaded successfully!")
        else:
            print("Dataset already exists.")
    
    def load_data(self, download_if_missing: bool = True) -> pd.DataFrame:
        """
        Load the heart disease dataset.
        
        Args:
            download_if_missing: Whether to download the dataset if it's missing
            
        Returns:
            DataFrame containing the heart disease data
        """
        if download_if_missing and not os.path.exists(self.dataset_path):
            self.download_dataset()
        
        # Load the data
        df = pd.read_csv(self.dataset_path, names=self.column_names, na_values='?')
        
        # Convert target to binary (0 = no disease, 1 = disease)
        df['target'] = (df['target'] > 0).astype(int)
        
        return df
    
    def get_feature_info(self) -> dict:
        """Return feature descriptions."""
        return self.feature_descriptions
    
    def basic_info(self, df: pd.DataFrame) -> None:
        """Print basic information about the dataset."""
        print("Dataset Shape:", df.shape)
        print("\nColumn Names:")
        for col in df.columns:
            print(f"  {col}: {self.feature_descriptions.get(col, 'Unknown')}")
        
        print(f"\nMissing Values:")
        print(df.isnull().sum())
        
        print(f"\nTarget Distribution:")
        print(df['target'].value_counts())
        
        print(f"\nBasic Statistics:")
        print(df.describe())

def main():
    """Main function to test the data loader."""
    loader = HeartDiseaseDataLoader()
    df = loader.load_data()
    loader.basic_info(df)

if __name__ == "__main__":
    main()

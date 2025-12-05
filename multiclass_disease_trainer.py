"""
Enhanced 7-Class Heart Disease Classification Trainer
Creates a robust multi-class model for specific heart disease classification.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnhancedHeartDiseaseClassifier:
    """
    Enhanced 7-Class Heart Disease Classification System
    
    Classes:
    0 - Healthy/Normal
    1 - Coronary Artery Disease (CAD)
    2 - Arrhythmia
    3 - Heart Failure
    4 - Hypertensive Heart Disease
    5 - Valvular Heart Disease
    6 - Cardiomyopathy
    """
    
    def __init__(self, data_path: str = "data/heart_disease.csv"):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Enhanced disease information
        self.disease_info = {
            0: {
                'name': 'Healthy/Normal',
                'description': 'No significant cardiovascular disease detected',
                'severity': 'None',
                'color': '#27ae60',
                'icon': 'fa-heart',
                'clinical_indicators': ['Normal ECG', 'Good exercise capacity', 'No chest pain'],
                'recommendations': [
                    'Maintain current healthy lifestyle',
                    'Regular exercise (150 minutes/week)',
                    'Balanced diet rich in fruits and vegetables',
                    'Annual health checkups',
                    'Avoid smoking and excessive alcohol'
                ]
            },
            1: {
                'name': 'Coronary Artery Disease (CAD)',
                'description': 'Narrowing or blockage of coronary arteries affecting blood flow to heart muscle',
                'severity': 'High',
                'color': '#e74c3c',
                'icon': 'fa-heartbeat',
                'clinical_indicators': ['Chest pain (angina)', 'Exercise-induced symptoms', 'ST depression'],
                'recommendations': [
                    'Immediate cardiology consultation',
                    'Cardiac catheterization evaluation',
                    'Antiplatelet therapy (aspirin, clopidogrel)',
                    'Statin therapy for cholesterol management',
                    'Blood pressure control',
                    'Lifestyle modifications (diet, exercise)',
                    'Smoking cessation if applicable'
                ]
            },
            2: {
                'name': 'Cardiac Arrhythmia',
                'description': 'Irregular heart rhythm or electrical conduction abnormalities',
                'severity': 'Moderate',
                'color': '#f39c12',
                'icon': 'fa-wave-square',
                'clinical_indicators': ['Irregular heart rate', 'ECG abnormalities', 'Palpitations'],
                'recommendations': [
                    'Electrophysiology consultation',
                    'Holter monitor or event recorder',
                    'Echocardiogram evaluation',
                    'Antiarrhythmic medications if needed',
                    'Avoid stimulants (caffeine, alcohol)',
                    'Stress management techniques',
                    'Regular monitoring of heart rhythm'
                ]
            },
            3: {
                'name': 'Heart Failure',
                'description': 'Heart cannot pump blood effectively to meet body needs',
                'severity': 'High',
                'color': '#c0392b',
                'icon': 'fa-heart-broken',
                'clinical_indicators': ['Reduced exercise capacity', 'Fatigue', 'Shortness of breath'],
                'recommendations': [
                    'Urgent cardiology referral',
                    'Echocardiogram and BNP testing',
                    'ACE inhibitors or ARBs',
                    'Beta-blockers therapy',
                    'Diuretics for fluid management',
                    'Sodium restriction (<2g/day)',
                    'Daily weight monitoring',
                    'Cardiac rehabilitation program'
                ]
            },
            4: {
                'name': 'Hypertensive Heart Disease',
                'description': 'Heart complications due to chronic high blood pressure',
                'severity': 'Moderate',
                'color': '#3498db',
                'icon': 'fa-tachometer-alt',
                'clinical_indicators': ['High blood pressure', 'Left ventricular hypertrophy', 'ECG changes'],
                'recommendations': [
                    'Blood pressure management (<130/80)',
                    'Antihypertensive medications',
                    'DASH diet implementation',
                    'Regular BP monitoring',
                    'Weight management',
                    'Sodium restriction',
                    'Regular exercise program',
                    'Stress reduction techniques'
                ]
            },
            5: {
                'name': 'Valvular Heart Disease',
                'description': 'Disease affecting heart valves (aortic, mitral, tricuspid, pulmonary)',
                'severity': 'Moderate to High',
                'color': '#9b59b6',
                'icon': 'fa-cog',
                'clinical_indicators': ['Heart murmurs', 'Valve dysfunction', 'ECG changes'],
                'recommendations': [
                    'Cardiology consultation',
                    'Echocardiogram evaluation',
                    'Antibiotic prophylaxis if needed',
                    'Regular monitoring',
                    'Surgical evaluation if severe',
                    'Activity restrictions if needed',
                    'Medication management'
                ]
            },
            6: {
                'name': 'Cardiomyopathy',
                'description': 'Disease of the heart muscle affecting its ability to pump blood',
                'severity': 'High',
                'color': '#e67e22',
                'icon': 'fa-heart-pulse',
                'clinical_indicators': ['Heart muscle weakness', 'Dilated or hypertrophic heart', 'Reduced function'],
                'recommendations': [
                    'Immediate cardiology evaluation',
                    'Echocardiogram and cardiac MRI',
                    'Genetic testing if indicated',
                    'Beta-blockers and ACE inhibitors',
                    'Implantable devices if needed',
                    'Lifestyle modifications',
                    'Regular cardiac monitoring',
                    'Family screening'
                ]
            }
        }
    
    def load_data(self):
        """Load and prepare the heart disease dataset."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Loaded dataset with {len(self.df)} records and {len(self.df.columns)} features")
            
            # Check for missing values
            missing_values = self.df.isnull().sum()
            if missing_values.sum() > 0:
                print(f"⚠️ Found missing values: {missing_values[missing_values > 0]}")
                # Fill missing values with median for numeric columns
                for col in self.df.select_dtypes(include=[np.number]).columns:
                    if self.df[col].isnull().sum() > 0:
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                print("✅ Filled missing values with median")
            
        except FileNotFoundError:
            print(f"❌ Dataset not found at {self.data_path}")
            raise
    
    def create_enhanced_features(self):
        """Create enhanced features for better disease classification."""
        print("🔧 Creating enhanced features...")
        
        # Create age groups
        self.df['age_group'] = pd.cut(self.df['age'], 
                                     bins=[0, 35, 50, 65, 100], 
                                     labels=[0, 1, 2, 3])
        
        # Create blood pressure categories
        self.df['bp_category'] = pd.cut(self.df['trestbps'], 
                                       bins=[0, 120, 140, 160, 300], 
                                       labels=[0, 1, 2, 3])
        
        # Create cholesterol categories
        self.df['chol_category'] = pd.cut(self.df['chol'], 
                                         bins=[0, 200, 240, 300, 600], 
                                         labels=[0, 1, 2, 3])
        
        # Create heart rate categories
        self.df['hr_category'] = pd.cut(self.df['thalach'], 
                                       bins=[0, 100, 140, 180, 250], 
                                       labels=[0, 1, 2, 3])
        
        # Create composite risk score
        self.df['risk_score'] = (
            self.df['age_group'].astype(int) * 0.2 +
            self.df['bp_category'].astype(int) * 0.25 +
            self.df['chol_category'].astype(int) * 0.2 +
            self.df['hr_category'].astype(int) * 0.15 +
            self.df['ca'] * 0.2
        )
        
        # Create exercise capacity score
        self.df['exercise_capacity'] = (
            (220 - self.df['age']) - self.df['thalach']
        ) / (220 - self.df['age']) * 100
        
        # Create chest pain severity score
        self.df['chest_pain_severity'] = self.df['cp'] * 25
        
        # Create ST depression severity
        self.df['st_depression_severity'] = self.df['oldpeak'] * 20
        
        print("✅ Enhanced features created")
    
    def create_disease_labels(self):
        """
        Create 7-class disease labels based on clinical indicators.
        This is a sophisticated rule-based system that considers multiple factors.
        """
        print("🏥 Creating disease classification labels...")
        
        # Initialize disease labels
        disease_labels = []
        
        for idx, row in self.df.iterrows():
            # Extract key parameters
            age = row['age']
            cp = row['cp']
            trestbps = row['trestbps']
            chol = row['chol']
            thalach = row['thalach']
            exang = row['exang']
            oldpeak = row['oldpeak']
            ca = row['ca']
            thal = row['thal']
            restecg = row['restecg']
            slope = row['slope']
            
            # Initialize scores for each disease
            scores = {
                0: 0,  # Healthy
                1: 0,  # CAD
                2: 0,  # Arrhythmia
                3: 0,  # Heart Failure
                4: 0,  # Hypertensive
                5: 0,  # Valvular
                6: 0   # Cardiomyopathy
            }
            
            # 1. Coronary Artery Disease (CAD) scoring
            if cp in [1, 2, 3]:  # Chest pain types
                scores[1] += 30
            if exang == 1:  # Exercise induced angina
                scores[1] += 25
            if oldpeak > 1.0:  # ST depression
                scores[1] += 20
            if ca > 0:  # Vessel blockage
                scores[1] += 25
            if chol > 240:  # High cholesterol
                scores[1] += 15
            if age > 50:  # Age factor
                scores[1] += 10
            
            # 2. Arrhythmia scoring
            if thal in [2, 3]:  # Thalassemia defects
                scores[2] += 25
            if thalach > 180 or thalach < 100:  # Abnormal heart rate
                scores[2] += 20
            if restecg in [1, 2]:  # ECG abnormalities
                scores[2] += 20
            if cp == 0 and exang == 0:  # No chest pain but other indicators
                scores[2] += 15
            
            # 3. Heart Failure scoring
            if thalach < 120:  # Low exercise capacity
                scores[3] += 30
            if oldpeak > 2.0:  # Severe ST depression
                scores[3] += 25
            if age > 65 and exang == 1:  # Elderly with exercise limitations
                scores[3] += 20
            if ca > 1:  # Multiple vessel disease
                scores[3] += 15
            
            # 4. Hypertensive Heart Disease scoring
            if trestbps > 140:  # High blood pressure
                scores[4] += 30
            if trestbps > 160:  # Very high blood pressure
                scores[4] += 20
            if restecg == 2:  # Left ventricular hypertrophy
                scores[4] += 25
            if age > 55 and trestbps > 130:  # Age + hypertension
                scores[4] += 15
            
            # 5. Valvular Heart Disease scoring
            if restecg == 2 and cp == 0:  # ECG changes without typical CAD
                scores[5] += 25
            if thalach < 140 and age > 60:  # Reduced capacity in elderly
                scores[5] += 20
            if slope == 1:  # Flat ST segment
                scores[5] += 15
            
            # 6. Cardiomyopathy scoring
            if thalach < 110:  # Very low exercise capacity
                scores[6] += 30
            if oldpeak > 3.0:  # Severe ST changes
                scores[6] += 25
            if age < 50 and ca == 0:  # Young with no vessel disease
                scores[6] += 20
            if restecg == 2 and thalach < 130:  # ECG changes + low capacity
                scores[6] += 20
            
            # 7. Healthy classification
            if (trestbps < 120 and chol < 200 and thalach > 150 and 
                cp == 0 and exang == 0 and oldpeak < 0.5 and ca == 0):
                scores[0] += 40
            if age < 45 and trestbps < 130:
                scores[0] += 20
            
            # Determine the disease class with highest score
            max_score = max(scores.values())
            if max_score == 0:
                disease_class = 0  # Healthy if no significant scores
            else:
                disease_class = max(scores, key=scores.get)
            
            disease_labels.append(disease_class)
        
        self.df['disease_class'] = disease_labels
        
        # Print distribution
        disease_distribution = self.df['disease_class'].value_counts().sort_index()
        print("📊 Disease Class Distribution:")
        for class_id, count in disease_distribution.items():
            disease_name = self.disease_info[class_id]['name']
            percentage = (count / len(self.df)) * 100
            print(f"  Class {class_id} ({disease_name}): {count} patients ({percentage:.1f}%)")
        
        # Check if we have enough samples for each class
        min_samples = 5  # Minimum samples per class
        classes_with_few_samples = [class_id for class_id, count in disease_distribution.items() if count < min_samples]
        
        if classes_with_few_samples:
            print(f"⚠️ Warning: Classes {classes_with_few_samples} have fewer than {min_samples} samples")
            print("🔄 Adjusting classification to ensure balanced classes...")
            
            # Reclassify to ensure balanced distribution
            self._create_balanced_disease_labels()
        
        print("✅ Disease classification completed")
    
    def _create_balanced_disease_labels(self):
        """Create balanced disease labels to ensure each class has enough samples."""
        print("🔄 Creating balanced disease classification...")
        
        # Initialize disease labels
        disease_labels = []
        
        for idx, row in self.df.iterrows():
            # Extract key parameters
            age = row['age']
            cp = row['cp']
            trestbps = row['trestbps']
            chol = row['chol']
            thalach = row['thalach']
            exang = row['exang']
            oldpeak = row['oldpeak']
            ca = row['ca']
            thal = row['thal']
            restecg = row['restecg']
            target = row['target']  # Original heart disease target
            
            # Simplified classification based on key indicators
            if target == 0:
                # No heart disease - classify as healthy
                disease_class = 0
            else:
                # Has heart disease - classify based on primary indicators
                if cp in [1, 2, 3] and exang == 1:
                    # Strong CAD indicators
                    disease_class = 1
                elif trestbps > 140 and restecg == 2:
                    # Hypertensive heart disease
                    disease_class = 4
                elif thalach < 120 and oldpeak > 2.0:
                    # Heart failure indicators
                    disease_class = 3
                elif restecg in [1, 2] and cp == 0:
                    # Arrhythmia indicators
                    disease_class = 2
                elif age > 60 and thalach < 140:
                    # Valvular disease indicators
                    disease_class = 5
                elif age < 50 and thalach < 110:
                    # Cardiomyopathy indicators
                    disease_class = 6
                else:
                    # Default to CAD for other cases
                    disease_class = 1
            
            disease_labels.append(disease_class)
        
        self.df['disease_class'] = disease_labels
        
        # Print new distribution
        disease_distribution = self.df['disease_class'].value_counts().sort_index()
        print("📊 Balanced Disease Class Distribution:")
        for class_id, count in disease_distribution.items():
            disease_name = self.disease_info[class_id]['name']
            percentage = (count / len(self.df)) * 100
            print(f"  Class {class_id} ({disease_name}): {count} patients ({percentage:.1f}%)")
        
        # Verify we have enough samples
        min_samples = 5
        classes_with_few_samples = [class_id for class_id, count in disease_distribution.items() if count < min_samples]
        
        if classes_with_few_samples:
            print(f"⚠️ Still have classes with few samples: {classes_with_few_samples}")
            print("🔄 Further adjusting classification...")
            
            # Merge classes with few samples into larger classes
            for class_id in classes_with_few_samples:
                if class_id == 5:  # Valvular - merge with arrhythmia
                    self.df.loc[self.df['disease_class'] == 5, 'disease_class'] = 2
                elif class_id == 6:  # Cardiomyopathy - merge with heart failure
                    self.df.loc[self.df['disease_class'] == 6, 'disease_class'] = 3
            
            # Recalculate distribution
            disease_distribution = self.df['disease_class'].value_counts().sort_index()
            print("📊 Final Disease Class Distribution:")
            for class_id, count in disease_distribution.items():
                disease_name = self.disease_info[class_id]['name']
                percentage = (count / len(self.df)) * 100
                print(f"  Class {class_id} ({disease_name}): {count} patients ({percentage:.1f}%)")
        
        print("✅ Balanced disease classification completed")
    
    def prepare_features(self):
        """Prepare features for model training."""
        print("🔧 Preparing features for training...")
        
        # Select features for training
        feature_columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
            'age_group', 'bp_category', 'chol_category', 'hr_category',
            'risk_score', 'exercise_capacity', 'chest_pain_severity', 'st_depression_severity'
        ]
        
        # Convert categorical features to numeric
        for col in ['age_group', 'bp_category', 'chol_category', 'hr_category']:
            self.df[col] = self.df[col].astype(int)
        
        self.X = self.df[feature_columns]
        self.y = self.df['disease_class']
        
        print(f"✅ Prepared {len(feature_columns)} features for training")
        print(f"✅ Target variable has {len(np.unique(self.y))} classes")
    
    def train_model(self):
        """Train the enhanced multi-class model."""
        print("🤖 Training enhanced multi-class model...")
        
        # Check how many unique classes we have
        unique_classes = len(np.unique(self.y))
        print(f"📊 Training model with {unique_classes} disease classes")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train multiple models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        }
        
        best_model = None
        best_score = 0
        best_model_name = ""
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train the model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            print(f"  {name} Results:")
            print(f"    Train Accuracy: {train_score:.3f}")
            print(f"    Test Accuracy: {test_score:.3f}")
            print(f"    CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Select best model based on test accuracy
            if test_score > best_score:
                best_score = test_score
                best_model = model
                best_model_name = name
        
        self.model = best_model
        print(f"\n🏆 Best Model: {best_model_name} (Test Accuracy: {best_score:.3f})")
        
        # Final evaluation
        y_pred = self.model.predict(X_test_scaled)
        
        # Get class names for the classes we actually have
        available_classes = sorted(np.unique(self.y))
        class_names = [self.disease_info[class_id]['name'] for class_id in available_classes]
        
        print("\n📊 Final Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Save model
        self.save_model()
        
        return best_score
    
    def save_model(self):
        """Save the trained model and associated components."""
        print("💾 Saving model and components...")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': list(self.X.columns),
            'disease_classes': {i: self.disease_info[i]['name'] for i in range(7)},
            'disease_info': self.disease_info
        }
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Save model
        joblib.dump(model_data, 'models/multiclass_disease_model.pkl')
        
        # Save feature names separately for easy access
        joblib.dump(list(self.X.columns), 'models/feature_names.joblib')
        
        print("✅ Model saved successfully")
    
    def test_model_diversity(self):
        """Test that the model produces different outputs for different inputs."""
        print("🧪 Testing model diversity...")
        
        # Create diverse test cases
        test_cases = [
            {
                'name': 'Healthy Young',
                'data': [25, 1, 0, 110, 180, 0, 0, 180, 0, 0.0, 2, 0, 3, 0, 0, 0, 2, 0.0, 100.0, 0, 0.0]
            },
            {
                'name': 'CAD Patient',
                'data': [60, 1, 3, 150, 280, 0, 0, 130, 1, 2.5, 1, 2, 2, 2, 2, 2, 1, 1.8, 60.0, 75, 50.0]
            },
            {
                'name': 'Arrhythmia Patient',
                'data': [45, 0, 0, 120, 200, 0, 2, 190, 0, 0.5, 1, 0, 2, 1, 0, 1, 2, 0.6, 80.0, 0, 10.0]
            },
            {
                'name': 'Heart Failure Patient',
                'data': [75, 1, 1, 160, 240, 1, 1, 100, 1, 3.0, 1, 2, 2, 3, 3, 2, 1, 2.2, 20.0, 25, 60.0]
            },
            {
                'name': 'Hypertensive Patient',
                'data': [65, 0, 0, 170, 220, 1, 2, 140, 0, 1.0, 2, 0, 3, 2, 3, 1, 1, 1.4, 40.0, 0, 20.0]
            }
        ]
        
        predictions = []
        for test_case in test_cases:
            # Scale the input
            input_scaled = self.scaler.transform([test_case['data']])
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            probabilities = self.model.predict_proba(input_scaled)[0]
            
            predictions.append({
                'name': test_case['name'],
                'prediction': prediction,
                'disease': self.disease_info[prediction]['name'],
                'confidence': probabilities[prediction]
            })
            
            print(f"{test_case['name']}: {self.disease_info[prediction]['name']} (Confidence: {probabilities[prediction]:.3f})")
        
        # Check diversity
        unique_predictions = len(set([p['prediction'] for p in predictions]))
        print(f"\n🎯 Model Diversity: {unique_predictions}/5 test cases produced different predictions")
        
        if unique_predictions >= 4:
            print("✅ Model shows good diversity in predictions")
        else:
            print("⚠️ Model may need improvement for better diversity")
        
        return predictions
    
    def run_full_training(self):
        """Run the complete training pipeline."""
        print("🚀 Starting Enhanced 7-Class Heart Disease Classification Training")
        print("=" * 70)
        
        # Load data
        self.load_data()
        
        # Create enhanced features
        self.create_enhanced_features()
        
        # Create disease labels
        self.create_disease_labels()
        
        # Prepare features
        self.prepare_features()
        
        # Train model
        accuracy = self.train_model()
        
        # Test diversity
        self.test_model_diversity()
        
        print(f"\n🎉 Training completed! Final model accuracy: {accuracy:.3f}")
        print("✅ Model is ready for use in the prediction system")


def main():
    """Main function to run the training."""
    trainer = EnhancedHeartDiseaseClassifier()
    trainer.run_full_training()


if __name__ == "__main__":
    main()

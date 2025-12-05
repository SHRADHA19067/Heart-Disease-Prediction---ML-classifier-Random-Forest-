"""
Simple Heart Disease Prediction Script
Provides easy-to-use prediction functionality.
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any, List

# Try to import from current directory first, then from relative imports
try:
    from data_loader import HeartDiseaseDataLoader
    from data_preprocessing import prepare_train_test_data
except ImportError:
    try:
        from .data_loader import HeartDiseaseDataLoader
        from .data_preprocessing import prepare_train_test_data
    except ImportError:
        # For standalone execution, import from src directory
        import sys
        sys.path.append(os.path.dirname(__file__))
        from data_loader import HeartDiseaseDataLoader
        from data_preprocessing import prepare_train_test_data

class SimpleHeartDiseasePredictor:
    """Simple prediction system for heart disease risk assessment."""
    
    def __init__(self, model_name: str = 'random_forest'):
        self.model_name = model_name
        self.model = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the trained model."""
        model_path = f'models/{self.model_name}_model.joblib'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        print(f"Loaded {self.model_name} model")
        
        # Set feature names based on basic features (13 features)
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
    
    def preprocess_patient_data(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess patient data for prediction."""
        # Extract basic features
        age = patient_data.get('age', 50)
        sex = patient_data.get('sex', 1)
        cp = patient_data.get('cp', 0)
        trestbps = patient_data.get('trestbps', 120)
        chol = patient_data.get('chol', 200)
        fbs = patient_data.get('fbs', 0)
        restecg = patient_data.get('restecg', 0)
        thalach = patient_data.get('thalach', 150)
        exang = patient_data.get('exang', 0)
        oldpeak = patient_data.get('oldpeak', 0.0)
        slope = patient_data.get('slope', 2)
        ca = patient_data.get('ca', 0)
        thal = patient_data.get('thal', 3)
        
        # Create enhanced features
        # Age groups
        if age < 35:
            age_group = 0
        elif age < 50:
            age_group = 1
        elif age < 65:
            age_group = 2
        else:
            age_group = 3
        
        # Blood pressure categories
        if trestbps < 120:
            bp_category = 0
        elif trestbps < 140:
            bp_category = 1
        elif trestbps < 160:
            bp_category = 2
        else:
            bp_category = 3
        
        # Cholesterol categories
        if chol < 200:
            chol_category = 0
        elif chol < 240:
            chol_category = 1
        elif chol < 300:
            chol_category = 2
        else:
            chol_category = 3
        
        # Heart rate categories
        if thalach < 100:
            hr_category = 0
        elif thalach < 140:
            hr_category = 1
        elif thalach < 180:
            hr_category = 2
        else:
            hr_category = 3
        
        # Composite risk score
        risk_score = (
            age_group * 0.2 +
            bp_category * 0.25 +
            chol_category * 0.2 +
            hr_category * 0.15 +
            ca * 0.2
        )
        
        # Exercise capacity score
        exercise_capacity = (
            (220 - age) - thalach
        ) / (220 - age) * 100 if age < 220 else 0
        
        # Chest pain severity score
        chest_pain_severity = cp * 25
        
        # ST depression severity
        st_depression_severity = oldpeak * 20
        
        # Create the feature array with 13 basic features (model was trained with 13)
        features = [
            age, sex, cp, trestbps, chol, fbs, restecg, 
            thalach, exang, oldpeak, slope, ca, thal
        ]
        
        return np.array([features])

    def _classify_cardiovascular_disease(self, patient_data: Dict[str, Any], risk_probability: float) -> Dict[str, Any]:
        """
        Enhanced 5-Class Cardiovascular Disease Classification System

        Classes:
        1. Healthy/Normal - No significant cardiovascular risk
        2. Coronary Artery Disease (CAD) - Blocked or narrowed arteries
        3. Arrhythmia - Irregular heart rhythm disorders
        4. Heart Failure - Heart cannot pump blood effectively
        5. Hypertensive Heart Disease - High blood pressure related heart problems
        """

        # Extract key parameters
        age = patient_data.get('age', 50)
        cp = patient_data.get('cp', 0)  # Chest pain type
        trestbps = patient_data.get('trestbps', 120)  # Resting blood pressure
        chol = patient_data.get('chol', 200)  # Cholesterol
        thalach = patient_data.get('thalach', 150)  # Max heart rate
        exang = patient_data.get('exang', 0)  # Exercise induced angina
        oldpeak = patient_data.get('oldpeak', 0)  # ST depression
        ca = patient_data.get('ca', 0)  # Number of major vessels
        thal = patient_data.get('thal', 3)  # Thalassemia

        # Initialize classification scores
        class_scores = {
            'healthy': 0,
            'coronary_artery_disease': 0,
            'arrhythmia': 0,
            'heart_failure': 0,
            'hypertensive_heart_disease': 0
        }

        # Classification Logic Based on Clinical Indicators

        # 1. Coronary Artery Disease (CAD) Indicators
        if cp in [1, 2, 3]:  # Typical/Atypical angina, Non-anginal pain
            class_scores['coronary_artery_disease'] += 25
        if exang == 1:  # Exercise induced angina
            class_scores['coronary_artery_disease'] += 20
        if oldpeak > 1.0:  # Significant ST depression
            class_scores['coronary_artery_disease'] += 15
        if ca > 0:  # Major vessels with blockage
            class_scores['coronary_artery_disease'] += 20
        if chol > 240:  # High cholesterol
            class_scores['coronary_artery_disease'] += 10

        # 2. Arrhythmia Indicators
        if thal in [6, 7]:  # Fixed or reversible defect
            class_scores['arrhythmia'] += 20
        if thalach > 180 or thalach < 100:  # Abnormal heart rate
            class_scores['arrhythmia'] += 15
        if cp == 4:  # Asymptomatic but with other indicators
            class_scores['arrhythmia'] += 10

        # 3. Heart Failure Indicators
        if thalach < 120:  # Low exercise capacity
            class_scores['heart_failure'] += 20
        if oldpeak > 2.0:  # Severe ST depression
            class_scores['heart_failure'] += 15
        if age > 65 and exang == 1:  # Elderly with exercise limitations
            class_scores['heart_failure'] += 15

        # 4. Hypertensive Heart Disease Indicators
        if trestbps > 140:  # High blood pressure
            class_scores['hypertensive_heart_disease'] += 25
        if trestbps > 160:  # Very high blood pressure
            class_scores['hypertensive_heart_disease'] += 15
        if age > 55 and trestbps > 130:  # Age + moderate hypertension
            class_scores['hypertensive_heart_disease'] += 10

        # 5. Healthy Classification
        if risk_probability < 0.2:
            class_scores['healthy'] += 30
        if trestbps < 120 and chol < 200 and thalach > 150:
            class_scores['healthy'] += 20
        if cp == 0 and exang == 0 and oldpeak < 0.5:
            class_scores['healthy'] += 15

        # Adjust scores based on overall risk probability
        risk_multiplier = min(risk_probability * 2, 1.5)
        for disease in ['coronary_artery_disease', 'arrhythmia', 'heart_failure', 'hypertensive_heart_disease']:
            class_scores[disease] = int(class_scores[disease] * risk_multiplier)

        # Determine primary classification
        primary_disease = max(class_scores, key=class_scores.get)
        confidence_score = class_scores[primary_disease]

        # Disease information mapping
        disease_info = {
            'healthy': {
                'name': 'Healthy/Normal',
                'description': 'No significant cardiovascular risk detected',
                'severity': 'None',
                'recommendations': [
                    'Maintain regular exercise routine',
                    'Continue healthy diet',
                    'Regular health checkups',
                    'Monitor blood pressure annually'
                ]
            },
            'coronary_artery_disease': {
                'name': 'Coronary Artery Disease (CAD)',
                'description': 'Blocked or narrowed coronary arteries affecting blood flow to heart muscle',
                'severity': 'Moderate to High',
                'recommendations': [
                    'Immediate cardiology consultation',
                    'Stress test or cardiac catheterization',
                    'Lifestyle modifications (diet, exercise)',
                    'Consider medication (statins, beta-blockers)',
                    'Smoking cessation if applicable'
                ]
            },
            'arrhythmia': {
                'name': 'Cardiac Arrhythmia',
                'description': 'Irregular heart rhythm or electrical conduction abnormalities',
                'severity': 'Mild to Moderate',
                'recommendations': [
                    'ECG monitoring and evaluation',
                    'Electrophysiology consultation if needed',
                    'Avoid excessive caffeine and stimulants',
                    'Monitor heart rate regularly',
                    'Consider antiarrhythmic medications'
                ]
            },
            'heart_failure': {
                'name': 'Heart Failure Risk',
                'description': 'Heart muscle weakness affecting pumping efficiency',
                'severity': 'High',
                'recommendations': [
                    'Urgent cardiology evaluation',
                    'Echocardiogram assessment',
                    'Fluid and sodium restriction',
                    'ACE inhibitors or ARBs consideration',
                    'Regular weight monitoring'
                ]
            },
            'hypertensive_heart_disease': {
                'name': 'Hypertensive Heart Disease',
                'description': 'Heart complications due to chronic high blood pressure',
                'severity': 'Moderate',
                'recommendations': [
                    'Blood pressure management',
                    'Antihypertensive medications',
                    'DASH diet implementation',
                    'Regular BP monitoring',
                    'Weight management'
                ]
            }
        }

        return {
            'primary_disease': primary_disease,
            'disease_name': disease_info[primary_disease]['name'],
            'description': disease_info[primary_disease]['description'],
            'severity': disease_info[primary_disease]['severity'],
            'confidence_score': confidence_score,
            'all_scores': class_scores,
            'recommendations': disease_info[primary_disease]['recommendations']
        }

    def predict_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a patient.
        
        Args:
            patient_data: Dictionary containing patient information
            
        Returns:
            Dictionary containing prediction results
        """
        # Validate input data
        required_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                           'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        missing_features = [f for f in required_features if f not in patient_data]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Preprocess the data
        X_processed = self.preprocess_patient_data(patient_data)
        
        # Make prediction
        prediction = self.model.predict(X_processed)[0]
        prediction_proba = self.model.predict_proba(X_processed)[0]
        
        # Calculate risk level
        risk_probability = prediction_proba[1]
        if risk_probability < 0.3:
            risk_level = "Low"
        elif risk_probability < 0.7:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        # Enhanced 5-Class Disease Classification
        disease_classification = self._classify_cardiovascular_disease(patient_data, risk_probability)

        return {
            'prediction': int(prediction),
            'prediction_text': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
            'risk_probability': float(risk_probability),
            'risk_level': risk_level,
            'confidence': float(max(prediction_proba)),
            'model_used': self.model_name,
            'disease_classification': disease_classification
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance if available."""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            print(f"Model {self.model_name} does not have feature importance")
            return None

def create_sample_patients() -> List[Dict[str, Any]]:
    """Create sample patient data for testing."""
    return [
        {
            'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
            'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
            'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
        },
        {
            'age': 37, 'sex': 1, 'cp': 2, 'trestbps': 130, 'chol': 250,
            'fbs': 0, 'restecg': 1, 'thalach': 187, 'exang': 0,
            'oldpeak': 3.5, 'slope': 0, 'ca': 0, 'thal': 2
        },
        {
            'age': 41, 'sex': 0, 'cp': 1, 'trestbps': 130, 'chol': 204,
            'fbs': 0, 'restecg': 0, 'thalach': 172, 'exang': 0,
            'oldpeak': 1.4, 'slope': 2, 'ca': 0, 'thal': 2
        },
        {
            'age': 56, 'sex': 1, 'cp': 0, 'trestbps': 120, 'chol': 236,
            'fbs': 0, 'restecg': 1, 'thalach': 178, 'exang': 0,
            'oldpeak': 0.8, 'slope': 2, 'ca': 0, 'thal': 2
        },
        {
            'age': 57, 'sex': 0, 'cp': 0, 'trestbps': 120, 'chol': 354,
            'fbs': 0, 'restecg': 1, 'thalach': 163, 'exang': 1,
            'oldpeak': 0.6, 'slope': 2, 'ca': 0, 'thal': 2
        }
    ]

def explain_features():
    """Explain what each feature means."""
    feature_explanations = {
        'age': 'Age in years',
        'sex': 'Sex (1 = male; 0 = female)',
        'cp': 'Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)',
        'trestbps': 'Resting blood pressure (in mm Hg on admission to the hospital)',
        'chol': 'Serum cholesterol in mg/dl',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
        'restecg': 'Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes; 0 = no)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)',
        'ca': 'Number of major vessels (0-3) colored by fluoroscopy',
        'thal': 'Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)'
    }
    
    print("Feature Explanations:")
    print("=" * 50)
    for feature, explanation in feature_explanations.items():
        print(f"{feature}: {explanation}")
    print()

def main():
    """Main function to demonstrate the prediction system."""
    print("Simple Heart Disease Prediction System")
    print("=" * 50)
    
    # Show feature explanations
    explain_features()
    
    # Initialize prediction system
    try:
        predictor = SimpleHeartDiseasePredictor(model_name='random_forest')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run model training first: python src/model_training.py")
        return
    
    # Create sample patients
    sample_patients = create_sample_patients()
    
    print("Testing with sample patients:")
    print("-" * 40)
    
    for i, patient in enumerate(sample_patients, 1):
        print(f"\nPatient {i}:")
        print(f"  Age: {patient['age']}, Sex: {'Male' if patient['sex'] == 1 else 'Female'}")
        print(f"  Chest Pain Type: {patient['cp']}, Resting BP: {patient['trestbps']}")
        print(f"  Cholesterol: {patient['chol']}, Max Heart Rate: {patient['thalach']}")
        print(f"  Exercise Angina: {'Yes' if patient['exang'] == 1 else 'No'}")
        print(f"  ST Depression: {patient['oldpeak']}")
        
        try:
            # Make prediction
            result = predictor.predict_patient(patient)
            
            print(f"  → Prediction: {result['prediction_text']}")
            print(f"  → Risk Probability: {result['risk_probability']:.1%}")
            print(f"  → Risk Level: {result['risk_level']}")
            print(f"  → Confidence: {result['confidence']:.1%}")
            
        except Exception as e:
            print(f"  → Error: {e}")
        
        print("-" * 50)
    
    # Show feature importance
    print("\nTop 10 Most Important Features:")
    print("-" * 35)
    importance_df = predictor.get_feature_importance()
    if importance_df is not None:
        for idx, row in importance_df.head(10).iterrows():
            print(f"{row['feature']:15} {row['importance']:.4f}")
    
    print(f"\nPrediction system demonstration completed!")
    print(f"Model used: {predictor.model_name}")

if __name__ == "__main__":
    main()

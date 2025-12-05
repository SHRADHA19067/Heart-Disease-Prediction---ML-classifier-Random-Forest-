"""
Improved Model Training Script for Heart Disease Prediction
Trains multiple models with hyperparameter tuning for maximum accuracy
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import os

print("="*70)
print("🚀 HEART DISEASE PREDICTION - IMPROVED MODEL TRAINING")


print("="*70)

# Load dataset
print("\n📂 Loading dataset...")
data_path = 'data/heart_disease_converted.csv'

if not os.path.exists(data_path):
    print(f"❌ Error: Dataset not found at {data_path}")
    print("Trying alternative dataset...")
    data_path = 'data/Heart_Disease_Prediction_3000_Updated.csv'
    
if not os.path.exists(data_path):
    print(f"❌ Error: No dataset found")
    exit(1)

df = pd.read_csv(data_path)
print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# Prepare data
print("\n🔧 Preparing data...")
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.joblib')
print("✅ Scaler saved")

print("\n" + "="*70)
print("🤖 TRAINING MODELS")
print("="*70)

# Model 1: Random Forest with GridSearch
print("\n1️⃣  Training Random Forest...")
rf_params = {
    'n_estimators': [200, 300, 500],
    'max_depth': [15, 20, 25, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
rf_grid.fit(X_train_scaled, y_train)

rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_cv_score = cross_val_score(rf_best, X_train_scaled, y_train, cv=10).mean()

print(f"   ✅ Best params: {rf_grid.best_params_}")
print(f"   📊 Test Accuracy: {rf_accuracy*100:.2f}%")
print(f"   📊 CV Score: {rf_cv_score*100:.2f}%")

# Model 2: Gradient Boosting
print("\n2️⃣  Training Gradient Boosting...")
gb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0]
}

gb = GradientBoostingClassifier(random_state=42)
gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
gb_grid.fit(X_train_scaled, y_train)

gb_best = gb_grid.best_estimator_
gb_pred = gb_best.predict(X_test_scaled)
gb_accuracy = accuracy_score(y_test, gb_pred)
gb_cv_score = cross_val_score(gb_best, X_train_scaled, y_train, cv=10).mean()

print(f"   ✅ Best params: {gb_grid.best_params_}")
print(f"   📊 Test Accuracy: {gb_accuracy*100:.2f}%")
print(f"   📊 CV Score: {gb_cv_score*100:.2f}%")

# Model 3: Ensemble (Voting Classifier)
print("\n3️⃣  Training Ensemble Model...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_best),
        ('gb', gb_best)
    ],
    voting='soft'
)

ensemble.fit(X_train_scaled, y_train)
ensemble_pred = ensemble.predict(X_test_scaled)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
ensemble_cv_score = cross_val_score(ensemble, X_train_scaled, y_train, cv=10).mean()

print(f"   📊 Test Accuracy: {ensemble_accuracy*100:.2f}%")
print(f"   📊 CV Score: {ensemble_cv_score*100:.2f}%")

# Select best model
print("\n" + "="*70)
print("📈 MODEL COMPARISON")
print("="*70)

models_comparison = {
    'Random Forest': {'accuracy': rf_accuracy, 'cv_score': rf_cv_score, 'model': rf_best},
    'Gradient Boosting': {'accuracy': gb_accuracy, 'cv_score': gb_cv_score, 'model': gb_best},
    'Ensemble': {'accuracy': ensemble_accuracy, 'cv_score': ensemble_cv_score, 'model': ensemble}
}

for name, metrics in models_comparison.items():
    print(f"\n{name}:")
    print(f"   Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"   CV Score: {metrics['cv_score']*100:.2f}%")

# Find best model
best_model_name = max(models_comparison, key=lambda x: models_comparison[x]['cv_score'])
best_model = models_comparison[best_model_name]['model']
best_accuracy = models_comparison[best_model_name]['accuracy']
best_cv_score = models_comparison[best_model_name]['cv_score']

print("\n" + "="*70)
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"   Test Accuracy: {best_accuracy*100:.2f}%")
print(f"   CV Score: {best_cv_score*100:.2f}%")
print("="*70)

# Detailed evaluation
print("\n📊 DETAILED EVALUATION")
print("="*70)

best_pred = best_model.predict(X_test_scaled)
best_pred_proba = best_model.predict_proba(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, best_pred, target_names=['No Disease', 'Disease']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_pred)
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# ROC AUC Score
roc_auc = roc_auc_score(y_test, best_pred_proba[:, 1])
print(f"\nROC AUC Score: {roc_auc:.4f}")

# Save best model
print("\n💾 Saving models...")
joblib.dump(best_model, 'models/random_forest_model.joblib')
joblib.dump(rf_best, 'models/random_forest_optimized.joblib')
joblib.dump(gb_best, 'models/gradient_boosting_optimized.joblib')
joblib.dump(ensemble, 'models/ensemble_model.joblib')

print("✅ Models saved successfully!")
print(f"   - models/random_forest_model.joblib (Best: {best_model_name})")
print(f"   - models/random_forest_optimized.joblib")
print(f"   - models/gradient_boosting_optimized.joblib")
print(f"   - models/ensemble_model.joblib")

print("\n" + "="*70)
print("✨ TRAINING COMPLETE!")
print(f"🎯 Final Model Accuracy: {best_accuracy*100:.2f}%")
print(f"🎯 Cross-Validation Score: {best_cv_score*100:.2f}%")
print("="*70)

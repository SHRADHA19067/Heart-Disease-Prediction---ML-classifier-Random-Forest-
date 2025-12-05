"""
Flask Web Application for Heart Disease Prediction
Provides a web interface with authentication for patients.
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import pandas as pd
import numpy as np
import joblib
import os
import sys
import traceback
import json
from datetime import datetime

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simple_prediction import SimpleHeartDiseasePredictor
from multiclass_predictor import MultiClassHeartDiseasePredictor
from models import db, User, Patient, Prediction, UserRole, init_db
from forms import LoginForm, PatientRegistrationForm, PredictionForm
# OTP removed - direct registration
from ml_visualizations import generate_ml_visualizations

app = Flask(__name__)
app.config['SECRET_KEY'] = 'heart_disease_prediction_secret_key_2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///heart_disease_prediction.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize database
init_db(app)

# Initialize the predictor
try:
    predictor = SimpleHeartDiseasePredictor(model_name='random_forest')
    print("✅ Heart Disease Prediction Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    predictor = None

# Initialize the multi-class predictor
try:
    multiclass_predictor = MultiClassHeartDiseasePredictor()
    print("✅ Multi-Class Disease Predictor Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading multi-class predictor: {e}")
    multiclass_predictor = None


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Landing page"""
    return render_template('landing_enhanced.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Patient login"""
    if current_user.is_authenticated:
        return redirect(url_for('patient_dashboard'))
    
    login_form = LoginForm()
    if login_form.validate_on_submit():
        user = User.query.filter_by(email=login_form.email.data).first()
        if user and user.check_password(login_form.password.data):
            if user.role == UserRole.PATIENT:
                login_user(user, remember=login_form.remember_me.data)
                flash('✅ Logged in successfully! Welcome back.', 'success')
                next_page = request.args.get('next')
                return redirect(next_page) if next_page else redirect(url_for('patient_dashboard'))
            else:
                flash('Invalid credentials for patient login.', 'danger')
        else:
            flash('Invalid email or password.', 'danger')
    
    return render_template('auth/auth_combined.html', login_form=login_form, register_form=PatientRegistrationForm(), is_login=True)


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Patient registration - direct signup without OTP"""
    if current_user.is_authenticated:
        return redirect(url_for('patient_dashboard'))
    
    register_form = PatientRegistrationForm()
    
    if register_form.validate_on_submit():
        # Check if user already exists
        existing_user = User.query.filter_by(email=register_form.email.data).first()
        if existing_user:
            flash('Email already registered. Please login.', 'warning')
            return redirect(url_for('login'))
        
        try:
            # Create user directly
            user = User(
                email=register_form.email.data,
                first_name=register_form.first_name.data,
                last_name=register_form.last_name.data,
                role=UserRole.PATIENT
            )
            user.set_password(register_form.password.data)
            
            db.session.add(user)
            db.session.flush()
            
            # Create patient profile
            patient = Patient(
                user_id=user.id,
                date_of_birth=register_form.date_of_birth.data,
                gender=register_form.gender.data,
                phone=register_form.phone.data,
                address=register_form.address.data,
                emergency_contact=register_form.emergency_contact.data,
                emergency_phone=register_form.emergency_phone.data,
                medical_history=register_form.medical_history.data,
                current_medications=register_form.current_medications.data,
                allergies=register_form.allergies.data,
                family_history=register_form.family_history.data
            )
            
            db.session.add(patient)
            db.session.commit()
            
            flash('✅ Signed up successfully! Please login to continue.', 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Registration failed: {str(e)}', 'danger')
            print(f"Registration error: {e}")
    
    return render_template('auth/auth_combined.html', login_form=LoginForm(), register_form=register_form, is_login=False)


@app.route('/logout')
@login_required
def logout():
    """Logout user"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/patient/dashboard')
@login_required
def patient_dashboard():
    """Patient dashboard with real-time data"""
    if current_user.role != UserRole.PATIENT:
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    
    patient = Patient.query.filter_by(user_id=current_user.id).first()
    predictions = Prediction.query.filter_by(patient_id=patient.id).order_by(Prediction.prediction_date.desc()).all()
    
    # Calculate real-time statistics from actual predictions
    total_predictions = len(predictions)
    
    if total_predictions > 0:
        # Calculate average risk
        avg_risk = sum(p.risk_probability for p in predictions) / total_predictions * 100
        
        # Count risk levels
        high_risk_count = sum(1 for p in predictions if p.risk_level == 'High')
        moderate_risk_count = sum(1 for p in predictions if p.risk_level == 'Moderate')
        low_risk_count = sum(1 for p in predictions if p.risk_level == 'Low')
        
        # Get latest prediction
        latest_prediction = predictions[0] if predictions else None
        
        # Calculate trend (comparing last 2 predictions)
        trend = 'stable'
        if len(predictions) >= 2:
            if predictions[0].risk_probability > predictions[1].risk_probability:
                trend = 'increasing'
            elif predictions[0].risk_probability < predictions[1].risk_probability:
                trend = 'decreasing'
    else:
        avg_risk = 0
        high_risk_count = 0
        moderate_risk_count = 0
        low_risk_count = 0
        latest_prediction = None
        trend = 'no_data'
    
    # Project details for dashboard
    project_details = {
        'dataset_name': 'UCI Heart Disease Dataset',
        'total_samples': 303,
        'model_accuracy': 93.44,
        'model_type': 'Random Forest Classifier',
        'positive_rate': 54.5
    }
    
    # Real-time dashboard stats
    dashboard_stats = {
        'total_predictions': total_predictions,
        'average_risk': round(avg_risk, 2),
        'high_risk_count': high_risk_count,
        'moderate_risk_count': moderate_risk_count,
        'low_risk_count': low_risk_count,
        'latest_prediction': latest_prediction,
        'trend': trend
    }
    
    return render_template('patient/dashboard.html', 
                         patient=patient, 
                         predictions=predictions, 
                         project_details=project_details,
                         dashboard_stats=dashboard_stats)


@app.route('/patient/predict', methods=['GET', 'POST'])
@login_required
def patient_predict():
    """Patient prediction form"""
    if current_user.role != UserRole.PATIENT:
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    
    form = PredictionForm()
    
    if form.validate_on_submit():
        try:
            # Prepare input data with proper type conversion
            input_data = {
                'age': int(form.age.data),
                'sex': int(form.sex.data),
                'cp': int(form.cp.data),
                'trestbps': int(form.trestbps.data),
                'chol': int(form.chol.data),
                'fbs': int(form.fbs.data),
                'restecg': int(form.restecg.data),
                'thalach': int(form.thalach.data),
                'exang': int(form.exang.data),
                'oldpeak': float(form.oldpeak.data),
                'slope': int(form.slope.data),
                'ca': int(form.ca.data),
                'thal': int(form.thal.data)
            }
            
            # Make prediction
            result = predictor.predict_patient(input_data)
            
            # Generate ML-focused visualizations
            graphs = generate_ml_visualizations(input_data, result, predictor)
            print(f"✅ Generated {len(graphs)} ML visualization graphs: {list(graphs.keys())}")
            
            # Save prediction to database
            patient = Patient.query.filter_by(user_id=current_user.id).first()
            prediction = Prediction(
                patient_id=patient.id,
                input_parameters=input_data,
                prediction_result=result['prediction_text'],
                risk_probability=result['risk_probability'],
                risk_level=result['risk_level'],
                model_confidence=result['confidence']
            )
            db.session.add(prediction)
            db.session.commit()
            
            return render_template('patient/prediction_result.html',
                                 result=result,
                                 input_data=input_data,
                                 graphs=graphs,
                                 prediction_id=prediction.id)
        
        except Exception as e:
            flash(f'Prediction error: {str(e)}', 'danger')
            traceback.print_exc()
    
    return render_template('patient/predict.html', form=form)


@app.route('/patient/history')
@login_required
def patient_history():
    """Patient prediction history"""
    if current_user.role != UserRole.PATIENT:
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    
    patient = Patient.query.filter_by(user_id=current_user.id).first()
    predictions = Prediction.query.filter_by(patient_id=patient.id).order_by(Prediction.prediction_date.desc()).all()
    
    return render_template('patient/history.html', predictions=predictions)


@app.route('/patient/visualizations')
@login_required
def enhanced_visualizations():
    """Enhanced visualizations page"""
    if current_user.role != UserRole.PATIENT:
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    
    # Redirect to predict page for now
    flash('Please make a prediction to see visualizations.', 'info')
    return redirect(url_for('patient_predict'))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Make prediction
        result = predictor.predict_patient(data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)

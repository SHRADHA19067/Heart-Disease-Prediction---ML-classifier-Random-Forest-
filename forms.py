"""
Forms for Heart Disease Prediction System
"""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SelectField, TextAreaField, DateField, IntegerField, BooleanField, ValidationError, SubmitField
from wtforms.validators import DataRequired, Email, Length, EqualTo, Optional, NumberRange
from models import User

class LoginForm(FlaskForm):
    """Login form for patients"""
    email = StringField('Email', validators=[
        DataRequired(message='Email is required'),
        Email(message='Please enter a valid email address')
    ], render_kw={'placeholder': 'Enter your email address'})
    
    password = PasswordField('Password', validators=[
        DataRequired(message='Password is required')
    ], render_kw={'placeholder': 'Enter your password'})
    
    remember_me = BooleanField('Remember Me')
    
    submit = SubmitField('Login')

# OTP removed: no OTP verification form

class PatientRegistrationForm(FlaskForm):
    """Registration form for patients"""
    # Basic Information
    first_name = StringField('First Name', validators=[
        DataRequired(message='First name is required'),
        Length(min=2, max=50, message='First name must be between 2 and 50 characters')
    ], render_kw={'placeholder': 'Enter your first name'})
    
    last_name = StringField('Last Name', validators=[
        DataRequired(message='Last name is required'),
        Length(min=2, max=50, message='Last name must be between 2 and 50 characters')
    ], render_kw={'placeholder': 'Enter your last name'})
    
    email = StringField('Email', validators=[
        DataRequired(message='Email is required'),
        Email(message='Please enter a valid email address')
    ], render_kw={'placeholder': 'Enter your email address'})
    
    password = PasswordField('Password', validators=[
        DataRequired(message='Password is required'),
        Length(min=6, message='Password must be at least 6 characters long')
    ], render_kw={'placeholder': 'Create a strong password'})
    
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(message='Please confirm your password'),
        EqualTo('password', message='Passwords must match')
    ], render_kw={'placeholder': 'Confirm your password'})
    
    # Personal Information
    date_of_birth = DateField('Date of Birth', validators=[
        Optional()
    ])
    
    gender = SelectField('Gender', choices=[
        ('', 'Select Gender'),
        ('male', 'Male'),
        ('female', 'Female'),
        ('other', 'Other')
    ], validators=[Optional()])
    
    phone = StringField('Phone Number', validators=[
        Optional(),
        Length(max=20, message='Phone number too long')
    ], render_kw={'placeholder': 'Enter your phone number'})
    
    address = TextAreaField('Address', validators=[
        Optional()
    ], render_kw={'placeholder': 'Enter your address', 'rows': 3})
    
    emergency_contact = StringField('Emergency Contact Name', validators=[
        Optional(),
        Length(max=100, message='Emergency contact name too long')
    ], render_kw={'placeholder': 'Emergency contact full name'})
    
    emergency_phone = StringField('Emergency Contact Phone', validators=[
        Optional(),
        Length(max=20, message='Emergency phone number too long')
    ], render_kw={'placeholder': 'Emergency contact phone'})
    
    # Medical Information
    medical_history = TextAreaField('Medical History', validators=[
        Optional()
    ], render_kw={'placeholder': 'Any relevant medical history, conditions, or surgeries', 'rows': 4})
    
    current_medications = TextAreaField('Current Medications', validators=[
        Optional()
    ], render_kw={'placeholder': 'List any medications you are currently taking', 'rows': 3})
    
    allergies = TextAreaField('Allergies', validators=[
        Optional()
    ], render_kw={'placeholder': 'List any known allergies', 'rows': 2})
    
    family_history = TextAreaField('Family Medical History', validators=[
        Optional()
    ], render_kw={'placeholder': 'Any relevant family medical history, especially heart conditions', 'rows': 3})
    
    # OTP removed
    
    def validate_email(self, field):
        """Check if email is already registered"""
        if User.query.filter_by(email=field.data.lower()).first():
            raise ValidationError('This email is already registered. Please use a different email or login.')

class PredictionForm(FlaskForm):
    """Form for heart disease prediction input"""
    # Basic Information
    age = IntegerField('Age (years)', validators=[
        DataRequired(message='Age is required'),
        NumberRange(min=1, max=120, message='Age must be between 1 and 120')
    ])
    
    sex = SelectField('Sex', choices=[
        ('', 'Select Sex'),
        ('1', 'Male'),
        ('0', 'Female')
    ], validators=[DataRequired(message='Sex is required')])
    
    cp = SelectField('Chest Pain Type', choices=[
        ('', 'Select Chest Pain Type'),
        ('0', 'Typical Angina'),
        ('1', 'Atypical Angina'),
        ('2', 'Non-Anginal Pain'),
        ('3', 'Asymptomatic')
    ], validators=[DataRequired(message='Chest pain type is required')])
    
    trestbps = IntegerField('Resting Blood Pressure (mm Hg)', validators=[
        DataRequired(message='Resting blood pressure is required'),
        NumberRange(min=80, max=250, message='Blood pressure must be between 80 and 250')
    ])
    
    chol = IntegerField('Cholesterol (mg/dl)', validators=[
        DataRequired(message='Cholesterol level is required'),
        NumberRange(min=100, max=600, message='Cholesterol must be between 100 and 600')
    ])
    
    fbs = SelectField('Fasting Blood Sugar > 120 mg/dl', choices=[
        ('', 'Select'),
        ('1', 'Yes (> 120 mg/dl)'),
        ('0', 'No (≤ 120 mg/dl)')
    ], validators=[DataRequired(message='Fasting blood sugar is required')])
    
    restecg = SelectField('Resting ECG Results', choices=[
        ('', 'Select ECG Result'),
        ('0', 'Normal'),
        ('1', 'ST-T Wave Abnormality'),
        ('2', 'Left Ventricular Hypertrophy')
    ], validators=[DataRequired(message='ECG result is required')])
    
    thalach = IntegerField('Maximum Heart Rate Achieved', validators=[
        DataRequired(message='Maximum heart rate is required'),
        NumberRange(min=60, max=220, message='Heart rate must be between 60 and 220')
    ])
    
    exang = SelectField('Exercise Induced Angina', choices=[
        ('', 'Select'),
        ('1', 'Yes'),
        ('0', 'No')
    ], validators=[DataRequired(message='Exercise induced angina is required')])
    
    oldpeak = StringField('ST Depression (Exercise vs Rest)', validators=[
        DataRequired(message='ST depression is required')
    ], render_kw={'placeholder': 'e.g., 0.0, 1.5, 2.3', 'step': '0.1', 'type': 'number'})
    
    slope = SelectField('Slope of Peak Exercise ST Segment', choices=[
        ('', 'Select Slope'),
        ('0', 'Upsloping'),
        ('1', 'Flat'),
        ('2', 'Downsloping')
    ], validators=[DataRequired(message='Slope is required')])
    
    ca = SelectField('Major Vessels Colored by Fluoroscopy', choices=[
        ('', 'Select Number'),
        ('0', '0'),
        ('1', '1'),
        ('2', '2'),
        ('3', '3')
    ], validators=[DataRequired(message='Number of major vessels is required')])
    
    thal = SelectField('Thalassemia', choices=[
        ('', 'Select Thalassemia Type'),
        ('0', 'Normal'),
        ('1', 'Fixed Defect'),
        ('2', 'Reversible Defect')
    ], validators=[DataRequired(message='Thalassemia type is required')])
    
    submit = SubmitField('Predict Heart Disease Risk')

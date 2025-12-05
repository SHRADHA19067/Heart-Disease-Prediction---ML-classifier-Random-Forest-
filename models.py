"""
Database models for Heart Disease Prediction System
Includes User authentication with Patient role
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import enum

db = SQLAlchemy()

class UserRole(enum.Enum):
    PATIENT = "patient"

class User(UserMixin, db.Model):
    """Base User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    role = db.Column(db.Enum(UserRole), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    patient_profile = db.relationship('Patient', backref='user', uselist=False, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if provided password matches hash"""
        return check_password_hash(self.password_hash, password)
    
    @property
    def full_name(self):
        """Get user's full name"""
        return f"{self.first_name} {self.last_name}"
    
    @property
    def is_patient(self):
        """Check if user is a patient"""
        return self.role == UserRole.PATIENT
    
    def __repr__(self):
        return f'<User {self.email} ({self.role.value})>'

class Patient(db.Model):
    """Patient profile with medical information"""
    __tablename__ = 'patients'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Personal Information
    date_of_birth = db.Column(db.Date)
    gender = db.Column(db.String(10))
    phone = db.Column(db.String(20))
    address = db.Column(db.Text)
    emergency_contact = db.Column(db.String(100))
    emergency_phone = db.Column(db.String(20))
    
    # Medical History
    medical_history = db.Column(db.Text)
    current_medications = db.Column(db.Text)
    allergies = db.Column(db.Text)
    family_history = db.Column(db.Text)
    
    # Relationships
    predictions = db.relationship('Prediction', backref='patient', lazy='dynamic', cascade='all, delete-orphan')
    
    @property
    def age(self):
        """Calculate age from date of birth"""
        if self.date_of_birth:
            today = datetime.now().date()
            return today.year - self.date_of_birth.year - ((today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day))
        return None
    
    def __repr__(self):
        return f'<Patient {self.user.full_name}>'

class Prediction(db.Model):
    """Store prediction results and history"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    
    # Prediction Data
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
    risk_probability = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    model_confidence = db.Column(db.Float, nullable=False)
    prediction_result = db.Column(db.String(50), nullable=False)
    
    # Input Parameters (stored as JSON for flexibility)
    input_parameters = db.Column(db.JSON, nullable=False)
    
    # Follow-up
    follow_up_required = db.Column(db.Boolean, default=False)
    follow_up_date = db.Column(db.DateTime)
    
    def __repr__(self):
        return f'<Prediction {self.patient.user.full_name} - {self.risk_level} ({self.prediction_date})>'

def init_db(app):
    """Initialize database with Flask app"""
    db.init_app(app)
    
    with app.app_context():
        # Create all tables
        db.create_all()
        
        print("✅ Database initialized successfully!")

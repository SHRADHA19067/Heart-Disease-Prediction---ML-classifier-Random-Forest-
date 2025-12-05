"""
Script to create dummy user for testing
"""
from app import app, db
from models import User, Patient, UserRole

def create_dummy_user():
    with app.app_context():
        # Check if user already exists
        existing_user = User.query.filter_by(email='test@example.com').first()
        if existing_user:
            print("✅ Dummy user already exists!")
            print("\n📧 Email: test@example.com")
            print("🔑 Password: Test@123")
            return
        
        # Create dummy user
        user = User(
            email='test@example.com',
            first_name='John',
            last_name='Doe',
            role=UserRole.PATIENT
        )
        user.set_password('Test@123')
        
        db.session.add(user)
        db.session.flush()
        
        # Create patient profile
        from datetime import date
        patient = Patient(
            user_id=user.id,
            date_of_birth=date(1979, 1, 1),  # Makes them ~45 years old
            phone='1234567890',
            gender='Male',
            address='123 Test Street, Test City'
        )
        
        db.session.add(patient)
        db.session.commit()
        
        print("✅ Dummy user created successfully!")
        print("\n" + "="*50)
        print("🎉 DUMMY USER CREDENTIALS")
        print("="*50)
        print("📧 Email: test@example.com")
        print("🔑 Password: Test@123")
        print("👤 Name: John Doe")
        print("📱 Phone: 1234567890")
        print("🎂 Date of Birth: 1979-01-01 (Age: ~45)")
        print("⚧ Gender: Male")
        print("="*50)
        print("\n✨ You can now login with these credentials!")
        print("\n🌐 Go to: http://localhost:5000/login")

if __name__ == '__main__':
    create_dummy_user()

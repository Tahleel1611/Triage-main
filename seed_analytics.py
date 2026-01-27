"""
Seed test data for analytics dashboard
"""
import sys
sys.path.insert(0, ".")

from datetime import datetime, timedelta, timezone
import random
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Use SQLite for dev
DATABASE_URL = "sqlite:///./triage_dev.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)

# Import models
from src.backend.models import Appointment, TriageResult, PriorityToken, Patient, User, UserRole, Base

# Create session
session = Session()

# First create a test user and patient if not exists
test_user = session.query(User).filter(User.email == "analytics_seed@test.com").first()
if not test_user:
    test_user = User(
        email="analytics_seed@test.com",
        hashed_password="$2b$12$fake_hash_for_seeding",
        role=UserRole.PATIENT,
    )
    session.add(test_user)
    session.flush()
    
    test_patient = Patient(
        user_id=test_user.id,
        first_name="Analytics",
        last_name="Seed",
    )
    session.add(test_patient)
    session.flush()
else:
    test_patient = session.query(Patient).filter(Patient.user_id == test_user.id).first()

# Create sample data
for i in range(20):
    # Random patient data
    age = random.randint(18, 85)
    esi_level = random.choices([1, 2, 3, 4, 5], weights=[5, 15, 30, 35, 15])[0]
    
    # Generate vitals
    if esi_level <= 2:
        hr = random.randint(100, 150)
        sbp = random.randint(70, 100)
        o2sat = random.randint(85, 94)
    else:
        hr = random.randint(60, 100)
        sbp = random.randint(100, 140)
        o2sat = random.randint(95, 100)
    
    vitals = {
        "Age": age,
        "Temp": round(random.uniform(97.5, 102.0), 1),
        "Pulse": hr,
        "Resp": random.randint(12, 24),
        "SBP": sbp,
        "DBP": random.randint(50, 90),
        "O2Sat": o2sat,
        "PainScale": random.randint(0, 10),
    }
    
    # Create appointment
    timestamp = datetime.now(timezone.utc) - timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
    
    appointment = Appointment(
        patient_id=test_patient.id,
        scheduled_time=timestamp,
        status="COMPLETED" if random.random() > 0.3 else "SCHEDULED",
    )
    session.add(appointment)
    session.flush()
    
    # Create triage result
    rl_actions = ["Assign to Bed", "Waiting Room", "Fast Track", "Observation"]
    
    # Generate fake SHAP values (11 features)
    shap_values = [round(random.uniform(-0.5, 0.5), 4) for _ in range(11)]
    
    triage_result = TriageResult(
        appointment_id=appointment.id,
        esi_level=esi_level,
        supervised_confidence=round(random.uniform(0.6, 0.98), 3),
        rl_action=random.choice(rl_actions),
        shap_values=json.dumps(shap_values),
        vitals=json.dumps(vitals),
        created_at=timestamp,
    )
    session.add(triage_result)
    session.flush()
    
    # Create priority token
    colors = {1: "RED", 2: "ORG", 3: "YEL", 4: "GRN", 5: "BLU"}
    token = PriorityToken(
        appointment_id=appointment.id,
        token_number=f"{colors[esi_level]}-{random.randint(1000, 9999)}",
        triage_level=esi_level,
        priority_score=100 - (esi_level * 15) + random.randint(-5, 5),
        estimated_wait_minutes=esi_level * random.randint(5, 15),
    )
    session.add(token)

session.commit()
session.close()

print(f"✅ Seeded 20 test patients into the database!")

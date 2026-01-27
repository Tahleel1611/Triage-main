import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from .database import Base


class UserRole(str, enum.Enum):
    PATIENT = "PATIENT"
    STAFF = "STAFF"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.PATIENT)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    patient_profile = relationship("Patient", back_populates="user", uselist=False)


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(Date, nullable=True)
    phone = Column(String(30), nullable=True)
    address = Column(Text, nullable=True)
    allergies = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="patient_profile")
    medical_history = relationship("MedicalHistory", back_populates="patient", cascade="all, delete-orphan")
    appointments = relationship("Appointment", back_populates="patient", cascade="all, delete-orphan")


class MedicalHistory(Base):
    __tablename__ = "medical_history"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    diagnosis = Column(Text, nullable=False)
    recorded_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    notes = Column(Text, nullable=True)

    patient = relationship("Patient", back_populates="medical_history")


class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    specialization = Column(String(150), nullable=False)
    department = Column(String(150), nullable=True)
    contact_email = Column(String(255), nullable=True, unique=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    appointments = relationship("Appointment", back_populates="doctor")


class AppointmentStatus(str, enum.Enum):
    SCHEDULED = "SCHEDULED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=True, index=True)
    scheduled_time = Column(DateTime(timezone=True), nullable=True)
    triage_level = Column(Integer, nullable=True)
    priority_score = Column(Integer, nullable=True)
    token_number = Column(String(20), nullable=True, unique=True)
    status = Column(Enum(AppointmentStatus), default=AppointmentStatus.SCHEDULED, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    patient = relationship("Patient", back_populates="appointments")
    doctor = relationship("Doctor", back_populates="appointments")
    triage_result = relationship("TriageResult", back_populates="appointment", uselist=False, cascade="all, delete-orphan")
    priority_token = relationship("PriorityToken", back_populates="appointment", uselist=False, cascade="all, delete-orphan")


class TriageResult(Base):
    __tablename__ = "triage_results"

    id = Column(Integer, primary_key=True)
    appointment_id = Column(Integer, ForeignKey("appointments.id"), nullable=False, unique=True)
    esi_level = Column(Integer, nullable=True)
    ktas_level = Column(Integer, nullable=True)
    model_version = Column(String(100), nullable=True)
    supervised_confidence = Column(Float, nullable=True)
    rl_action = Column(String(50), nullable=True)
    rl_policy_version = Column(String(100), nullable=True)
    bert_embedding = Column(LargeBinary, nullable=True)
    shap_values = Column(JSON, nullable=True)
    vitals = Column(JSON, nullable=True)  # Store vitals for clinical insight calculations
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    appointment = relationship("Appointment", back_populates="triage_result")


class PriorityToken(Base):
    __tablename__ = "priority_tokens"
    __table_args__ = (UniqueConstraint("token_number", name="uq_priority_token"),)

    id = Column(Integer, primary_key=True)
    appointment_id = Column(Integer, ForeignKey("appointments.id"), nullable=False, unique=True)
    token_number = Column(String(20), nullable=False)
    triage_level = Column(Integer, nullable=False)
    priority_score = Column(Integer, nullable=False)
    estimated_wait_minutes = Column(Integer, nullable=True)
    issued_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    status = Column(String(50), default="ACTIVE", nullable=False)

    appointment = relationship("Appointment", back_populates="priority_token")

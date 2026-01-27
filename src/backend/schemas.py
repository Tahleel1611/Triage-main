from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field

from .models import UserRole, AppointmentStatus


# --- Auth ---
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    sub: str
    role: UserRole
    exp: int


class UserBase(BaseModel):
    email: EmailStr
    role: UserRole = UserRole.PATIENT


class UserCreate(UserBase):
    password: str = Field(min_length=8)


class UserOut(UserBase):
    id: int
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Patient & Doctor ---
class PatientBase(BaseModel):
    first_name: str
    last_name: str
    date_of_birth: Optional[datetime] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    allergies: Optional[str] = None


class PatientCreate(PatientBase):
    user_id: int


class PatientOut(PatientBase):
    id: int
    user_id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class DoctorBase(BaseModel):
    first_name: str
    last_name: str
    specialization: str
    department: Optional[str] = None
    contact_email: Optional[EmailStr] = None


class DoctorCreate(DoctorBase):
    pass


class DoctorOut(DoctorBase):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Appointments & Triage ---
class AppointmentBase(BaseModel):
    patient_id: int
    doctor_id: Optional[int] = None
    scheduled_time: Optional[datetime] = None
    triage_level: Optional[int] = None
    priority_score: Optional[int] = None
    token_number: Optional[str] = None
    status: AppointmentStatus = AppointmentStatus.SCHEDULED


class AppointmentCreate(AppointmentBase):
    pass


class AppointmentOut(AppointmentBase):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class TriageRequest(BaseModel):
    chief_complaint: str = Field(min_length=1, max_length=5000)
    static_vitals: Dict[str, float]
    time_series_vitals: Optional[List[Dict[str, Any]]] = None
    arrival_mode: Optional[str] = None
    
    @classmethod
    def validate_chief_complaint(cls, v):
        if not v or not v.strip():
            return "General complaint"  # Default for empty
        return v.strip()[:5000]  # Truncate if too long


class TriageResponse(BaseModel):
    appointment_id: int
    triage_level: int
    priority_score: int
    token_number: str
    estimated_wait_minutes: Optional[int] = None
    action: Optional[str] = None


class PriorityTokenOut(BaseModel):
    token_number: str
    triage_level: int
    priority_score: int
    estimated_wait_minutes: Optional[int] = None
    issued_at: datetime

    model_config = {"from_attributes": True}

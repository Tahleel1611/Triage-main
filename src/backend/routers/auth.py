from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from .. import models, schemas, security
from ..database import get_db
from ..deps import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


def authenticate_user(db: Session, email: str, password: str) -> models.User | None:
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        return None
    if not security.verify_password(password, user.hashed_password):
        return None
    return user


@router.post("/register", response_model=schemas.UserOut)
def register_user(payload: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    user = models.User(
        email=payload.email,
        hashed_password=security.get_password_hash(payload.password),
        role=payload.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/token", response_model=schemas.Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")

    access_token = security.create_access_token(subject=str(user.id), role=user.role.value)
    return schemas.Token(access_token=access_token)


@router.get("/me")
def get_current_user_info(
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get current user info including patient profile if exists."""
    result = {
        "id": user.id,
        "email": user.email,
        "role": user.role.value,
        "patient_profile": None
    }
    if user.patient_profile:
        result["patient_profile"] = {
            "id": user.patient_profile.id,
            "first_name": user.patient_profile.first_name,
            "last_name": user.patient_profile.last_name,
        }
    return result


@router.post("/patient-profile", response_model=schemas.PatientOut)
def create_patient_profile(
    payload: schemas.PatientBase,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a patient profile for the current user."""
    if user.patient_profile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Patient profile already exists"
        )
    
    patient = models.Patient(
        user_id=user.id,
        first_name=payload.first_name,
        last_name=payload.last_name,
        date_of_birth=payload.date_of_birth,
        phone=payload.phone,
        address=payload.address,
        allergies=payload.allergies,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient

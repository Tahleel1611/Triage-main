from datetime import datetime
from typing import Any

import asyncio
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session
import numpy as np

from .. import models, schemas
from ..database import SessionLocal, get_db
from ..deps import get_current_user, require_role
from ..services.priority import issue_priority_token, build_ed_state, get_operational_action
from ..services.events import event_broker
from ..services import inference

router = APIRouter(prefix="/triage", tags=["triage"])


FEATURE_NAMES = [
    "Age", "Temp", "Pulse", "Resp", "SBP", "DBP", "O2Sat", "PainScale", "ArrivalMode",
] + [f"BERT_{i}" for i in range(768)]


def _extract_top_shap(shap_payload: Any, top_n: int = 3):
    """Extract top N SHAP features by absolute value."""
    try:
        if isinstance(shap_payload, list) and len(shap_payload) > 0:
            values = shap_payload[0] if isinstance(shap_payload[0], list) else shap_payload
        else:
            values = shap_payload
        if not values:
            return []
        arr = np.array(values).flatten()
        indices = np.argsort(np.abs(arr))[-top_n:][::-1]
        top_features = []
        for idx in indices:
            name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"Feature_{idx}"
            top_features.append({"feature": name, "impact": float(arr[idx])})
        return top_features
    except Exception:
        return []


def _persist_shap_async(appointment_id: int, shap_payload: Any):
    db = SessionLocal()
    try:
        triage_result = (
            db.query(models.TriageResult)
            .filter(models.TriageResult.appointment_id == appointment_id)
            .one_or_none()
        )
        if triage_result is None:
            return
        triage_result.shap_values = shap_payload
        db.add(triage_result)
        db.commit()
        top_shap = _extract_top_shap(shap_payload)
        asyncio.run(event_broker.publish({
            "event_type": "shap_ready",
            "appointment_id": appointment_id,
            "shap_ready": True,
            "top_features": top_shap,
        }))
    finally:
        db.close()


@router.post("/assess", response_model=schemas.TriageResponse)
def triage_assessment(
    payload: schemas.TriageRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    patient_id = None
    if user.role == models.UserRole.PATIENT:
        if not user.patient_profile:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No patient profile found")
        patient_id = user.patient_profile.id
    else:
        patient_id = payload.static_vitals.get("patient_id") if payload.static_vitals else None

    if not patient_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="patient_id is required")

    patient = db.get(models.Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

    app_state = request.app.state
    
    # Check if models are loaded (support dev mode with mock models)
    if not hasattr(app_state, "sup_model") or not hasattr(app_state, "preprocessor"):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Models not loaded")

    # --- Inference pipeline with error handling ---
    try:
        # In dev mode, BERT model may be None - use mock embedding
        if app_state.tokenizer is not None and app_state.bert_model is not None:
            embedding = inference.get_bert_embedding(payload.chief_complaint, app_state.tokenizer, app_state.bert_model, app_state.device)
        else:
            # Dev mode: mock embedding (768-dim like ClinicalBERT)
            embedding = np.random.randn(768).astype(np.float32)
        
        structured_df = inference.prepare_structured_features(payload.static_vitals, payload.arrival_mode)
        X_combined = inference.combine_features(app_state.preprocessor, structured_df, embedding)
        triage_level, sup_conf, probs = inference.predict_sup(app_state.sup_model, X_combined)
    except Exception as e:
        # Log the error and fallback to a safe default
        import logging
        logging.error(f"Inference pipeline error: {e}")
        # Fallback: assign middle acuity (KTAS 3 - Urgent) with low confidence
        triage_level = 3
        sup_conf = 0.5
        probs = np.array([0.1, 0.15, 0.5, 0.15, 0.1])
        embedding = np.zeros(768, dtype=np.float32)
    
    arrival_time = datetime.utcnow()

    # Database operations with retry for SQLite locking
    max_retries = 3
    for attempt in range(max_retries):
        try:
            appointment = models.Appointment(patient_id=patient.id, triage_level=triage_level, scheduled_time=arrival_time)
            db.add(appointment)
            db.commit()
            db.refresh(appointment)

            triage_result = models.TriageResult(
                appointment_id=appointment.id,
                esi_level=triage_level,
                supervised_confidence=sup_conf,
                model_version="v1",
                bert_embedding=embedding.astype(np.float32).tobytes(),
                vitals=payload.static_vitals,  # Store vitals for clinical insights
            )
            db.add(triage_result)
            db.commit()
            break
        except Exception as db_error:
            db.rollback()
            if attempt < max_retries - 1:
                import time
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                continue
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database busy, please retry")

    action = None
    if hasattr(app_state, "rl_model") and app_state.rl_model is not None:
        try:
            ed_state = build_ed_state(probs, payload.static_vitals, db)
            action = get_operational_action(app_state.rl_model, ed_state)
            triage_result.rl_action = action
            db.add(triage_result)
            db.commit()
        except Exception as rl_error:
            import logging
            logging.error(f"RL action error: {rl_error}")
            # Continue without RL action

    token = issue_priority_token(db, appointment=appointment, triage_level=triage_level, arrival_time=arrival_time)

    # Update appointment status based on action
    if action:
        if action == "Assign to Bed" or action == "Direct to Fast Track":
            appointment.status = models.AppointmentStatus.IN_PROGRESS
        else:
            appointment.status = models.AppointmentStatus.SCHEDULED
        db.add(appointment)
        db.commit()

    background_tasks.add_task(
        event_broker.publish,
        {
            "event_type": "new_patient",
            "appointment_id": appointment.id,
            "token": token.token_number,
            "triage_level": triage_level,
            "priority_score": token.priority_score,
            "action": action,
            "vitals": {
                "SBP": payload.static_vitals.get("SBP"),
                "HR": payload.static_vitals.get("Pulse") or payload.static_vitals.get("HR"),
                "RR": payload.static_vitals.get("Resp") or payload.static_vitals.get("RR"),
                "O2Sat": payload.static_vitals.get("O2Sat"),
            },
            "chief_complaint": payload.chief_complaint[:100],
            "shap_ready": False,
        },
    )

    # Heavy SHAP computation in background
    shap_values = inference.compute_shap_values(app_state.sup_model, X_combined)
    if shap_values is not None:
        if isinstance(shap_values, list):
            shap_payload = [sv.tolist() for sv in shap_values]
        else:
            shap_payload = shap_values.tolist()
        background_tasks.add_task(_persist_shap_async, appointment_id=appointment.id, shap_payload=shap_payload)

    response = schemas.TriageResponse(
        appointment_id=appointment.id,
        triage_level=triage_level,
        priority_score=token.priority_score,
        token_number=token.token_number,
        estimated_wait_minutes=token.estimated_wait_minutes,
        action=action,
    )
    return response


@router.get("/me/appointments", response_model=list[schemas.AppointmentOut])
def my_appointments(
    db: Session = Depends(get_db),
    user: models.User = Depends(require_role(models.UserRole.PATIENT)),
):
    patient = user.patient_profile
    if not patient:
        return []
    return patient.appointments


@router.get("/result/{appointment_id}")
def get_triage_result(
    appointment_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Fetch triage result with SHAP values for a specific appointment."""
    triage = (
        db.query(models.TriageResult)
        .filter(models.TriageResult.appointment_id == appointment_id)
        .one_or_none()
    )
    if not triage:
        raise HTTPException(status_code=404, detail="Triage result not found")

    # Extract top SHAP features if available
    top_features = []
    if triage.shap_values:
        top_features = _extract_top_shap(triage.shap_values, top_n=5)

    # Calculate Shock Index (HR/SBP) for clinical insights
    shock_index = None
    if triage.vitals:
        hr = triage.vitals.get("Pulse") or triage.vitals.get("HR")
        sbp = triage.vitals.get("SBP")
        if hr and sbp and sbp > 0:
            shock_index = round(hr / sbp, 2)

    return {
        "appointment_id": triage.appointment_id,
        "esi_level": triage.esi_level,
        "ktas_level": triage.ktas_level,
        "supervised_confidence": triage.supervised_confidence,
        "rl_action": triage.rl_action,
        "model_version": triage.model_version,
        "shap_values": triage.shap_values,
        "top_features": top_features,
        "shock_index": shock_index,
        "vitals": triage.vitals,
        "created_at": triage.created_at.isoformat() if triage.created_at else None,
    }

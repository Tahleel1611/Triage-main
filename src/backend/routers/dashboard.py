import asyncio
from typing import AsyncGenerator, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..deps import require_role
from ..services.events import event_broker

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


async def event_stream() -> AsyncGenerator[str, None]:
    queue = await event_broker.subscribe()
    try:
        while True:
            event = await queue.get()
            yield event_broker.format_sse(event)
    finally:
        await event_broker.unsubscribe(queue)


@router.get("/stream")
async def stream_dashboard():
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/active-patients")
def get_active_patients(
    db: Session = Depends(get_db),
    _user: models.User = Depends(require_role(models.UserRole.STAFF)),
):
    """Fetch all active (SCHEDULED/IN_PROGRESS) appointments with triage data."""
    appointments = (
        db.query(models.Appointment)
        .filter(
            models.Appointment.status.in_([
                models.AppointmentStatus.SCHEDULED,
                models.AppointmentStatus.IN_PROGRESS,
            ])
        )
        .order_by(models.Appointment.priority_score.desc())
        .all()
    )

    result = []
    for appt in appointments:
        triage = appt.triage_result
        patient = appt.patient
        result.append({
            "appointment_id": appt.id,
            "token": appt.token_number,
            "triage_level": appt.triage_level,
            "priority_score": appt.priority_score,
            "status": appt.status.value,
            "action": triage.rl_action if triage else None,
            "shap_ready": triage.shap_values is not None if triage else False,
            "patient_name": f"{patient.first_name} {patient.last_name}" if patient else None,
        })
    return result


@router.get("/triage-result/{appointment_id}")
def get_triage_result(
    appointment_id: int,
    db: Session = Depends(get_db),
    _user: models.User = Depends(require_role(models.UserRole.STAFF)),
):
    """Fetch full triage result including SHAP values for a specific appointment."""
    triage = (
        db.query(models.TriageResult)
        .filter(models.TriageResult.appointment_id == appointment_id)
        .one_or_none()
    )
    if not triage:
        raise HTTPException(status_code=404, detail="Triage result not found")

    return {
        "appointment_id": triage.appointment_id,
        "esi_level": triage.esi_level,
        "ktas_level": triage.ktas_level,
        "supervised_confidence": triage.supervised_confidence,
        "rl_action": triage.rl_action,
        "model_version": triage.model_version,
        "shap_values": triage.shap_values,
        "created_at": triage.created_at.isoformat() if triage.created_at else None,
    }

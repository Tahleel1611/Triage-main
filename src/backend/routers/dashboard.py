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


@router.get("/handoff", response_model=schemas.HandoffReport)
def generate_handoff_report(
    db: Session = Depends(get_db),
    _user: models.User = Depends(require_role(models.UserRole.STAFF)),
):
    """
    Generate an AI-powered shift handoff report.
    Summarizes all active patients with trend analysis and critical alerts.
    """
    from datetime import datetime, timezone
    import json

    # Fetch all active appointments with triage data
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

    now = datetime.now(timezone.utc)
    patients = []
    alerts = []
    critical_count = 0
    urgent_count = 0
    stable_count = 0
    total_wait = 0
    wait_count = 0

    for appt in appointments:
        triage = appt.triage_result
        patient = appt.patient
        if not patient:
            continue

        # Count by acuity
        level = appt.triage_level or 5
        if level <= 2:
            critical_count += 1
        elif level == 3:
            urgent_count += 1
        else:
            stable_count += 1

        # Calculate time in ED
        time_in_ed = None
        if appt.created_at:
            created = appt.created_at
            if created.tzinfo is None:
                from datetime import timezone as tz
                created = created.replace(tzinfo=tz.utc)
            time_in_ed = int((now - created).total_seconds() / 60)
            total_wait += time_in_ed
            wait_count += 1

        # Parse vitals and calculate shock index
        vitals = {}
        shock_index = None
        vitals_summary = "No vitals recorded"
        trend_status = "stable"

        if triage and triage.vitals:
            vitals = triage.vitals
            if isinstance(vitals, str):
                try:
                    vitals = json.loads(vitals)
                except (json.JSONDecodeError, TypeError):
                    vitals = {}

            hr = vitals.get("HR") or vitals.get("Pulse")
            sbp = vitals.get("SBP")
            o2 = vitals.get("O2Sat")
            rr = vitals.get("RR") or vitals.get("Resp")

            # Vitals summary string
            parts = []
            if hr:
                parts.append(f"HR {int(hr)}")
            if sbp:
                parts.append(f"BP {int(sbp)}")
            if o2:
                parts.append(f"O2 {int(o2)}%")
            if rr:
                parts.append(f"RR {int(rr)}")
            vitals_summary = " | ".join(parts) if parts else "No vitals"

            # Shock index
            if hr and sbp and sbp > 0:
                shock_index = round(hr / sbp, 2)
                if shock_index > 1.0:
                    trend_status = "deteriorating"
                    alerts.append(f"⚠️ {patient.first_name} {patient.last_name}: Critical Shock Index ({shock_index})")
                elif shock_index > 0.7:
                    trend_status = "deteriorating"

            # Check for critical vital signs
            if hr and hr > 120:
                alerts.append(f"🔴 {patient.first_name} {patient.last_name}: Tachycardia (HR {int(hr)})")
            if sbp and sbp < 90:
                alerts.append(f"🔴 {patient.first_name} {patient.last_name}: Hypotension (SBP {int(sbp)})")
            if o2 and o2 < 92:
                alerts.append(f"🔴 {patient.first_name} {patient.last_name}: Hypoxia (O2 {int(o2)}%)")

        handoff_patient = schemas.HandoffPatient(
            appointment_id=appt.id,
            patient_id=patient.id,
            patient_name=f"{patient.first_name} {patient.last_name}",
            token=appt.token_number or "N/A",
            triage_level=level,
            chief_complaint=triage.rl_action if triage and triage.rl_action else "Assessment pending",
            vitals_summary=vitals_summary,
            trend_status=trend_status,
            shock_index=shock_index,
            time_in_ed=time_in_ed,
            action=triage.rl_action if triage else None,
        )
        patients.append(handoff_patient)

    # Generate AI summary
    summary_parts = []
    if critical_count > 0:
        summary_parts.append(f"{critical_count} critical patient(s) requiring immediate attention")
    if urgent_count > 0:
        summary_parts.append(f"{urgent_count} urgent case(s) in queue")
    if stable_count > 0:
        summary_parts.append(f"{stable_count} stable patient(s) for routine care")
    
    if not summary_parts:
        summary = "ED is currently clear. No active patients."
    else:
        summary = ". ".join(summary_parts) + "."

    # Add specific concerns
    deteriorating = [p for p in patients if p.trend_status == "deteriorating"]
    if deteriorating:
        names = ", ".join([p.patient_name for p in deteriorating[:3]])
        summary += f" Watch closely: {names}."

    avg_wait = round(total_wait / wait_count, 1) if wait_count > 0 else None

    # Watch list = critical + deteriorating
    watch_list = [p for p in patients if p.triage_level <= 2 or p.trend_status == "deteriorating"]

    return schemas.HandoffReport(
        generated_at=now,
        total_patients=len(patients),
        critical_count=critical_count,
        urgent_count=urgent_count,
        stable_count=stable_count,
        avg_wait_minutes=avg_wait,
        summary=summary,
        alerts=alerts[:10],  # Limit to top 10 alerts
        watch_list=watch_list,
        all_patients=patients,
    )

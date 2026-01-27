from datetime import datetime
from typing import Optional
import random

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from .. import models

COLOR_MAP = {
    1: "RED",
    2: "ORG",
    3: "YEL",
    4: "GRN",
    5: "BLU",
}

ACTION_MAP = {
    0: "Assign to Bed",
    1: "Move to Waiting Room",
    2: "Direct to Fast Track",
}


def calculate_priority_score(triage_level: int, arrival_time: datetime) -> int:
    minutes = int(arrival_time.timestamp() // 60)
    return (6 - triage_level) * 100000 + minutes


def generate_token(triage_level: int, sequence: int) -> str:
    prefix = COLOR_MAP.get(triage_level, "UNK")
    return f"{prefix}-{sequence:04d}"


def issue_priority_token(
    db: Session,
    appointment: models.Appointment,
    triage_level: int,
    arrival_time: Optional[datetime] = None,
    estimated_wait_minutes: Optional[int] = None,
) -> models.PriorityToken:
    arrival = arrival_time or datetime.utcnow()
    score = calculate_priority_score(triage_level, arrival)
    
    # Generate unique token with retry on collision
    max_retries = 5
    for attempt in range(max_retries):
        # Use random suffix to avoid collisions
        rand_suffix = random.randint(1000, 9999)
        token_value = generate_token(triage_level, sequence=rand_suffix)
        
        token = models.PriorityToken(
            appointment_id=appointment.id,
            triage_level=triage_level,
            priority_score=score,
            token_number=token_value,
            estimated_wait_minutes=estimated_wait_minutes,
        )
        appointment.triage_level = triage_level
        appointment.priority_score = score
        appointment.token_number = token_value

        try:
            db.add(token)
            db.flush()  # Try to insert, will fail if duplicate
            db.commit()
            db.refresh(token)
            return token
        except IntegrityError:
            db.rollback()
            if attempt == max_retries - 1:
                raise
            continue
    
    # Should never reach here
    raise RuntimeError("Failed to generate unique token")


def _compute_queue_stats(db: Session):
    waiting = db.query(models.Appointment).filter(models.Appointment.status == models.AppointmentStatus.SCHEDULED)
    count_waiting = waiting.count()

    avg_wait_minutes = waiting.with_entities(
        func.avg(func.extract("epoch", func.now() - models.Appointment.created_at) / 60.0)
    ).scalar()
    avg_wait_minutes = float(avg_wait_minutes) if avg_wait_minutes is not None else 0.0
    return count_waiting, avg_wait_minutes


def build_ed_state(probs: np.ndarray, vitals: dict[str, float], db: Session) -> np.ndarray:
    queue_len, avg_wait = _compute_queue_stats(db)
    risk = 0.9 if vitals.get("O2Sat", 100) < 90 or vitals.get("SBP", 120) < 90 else 0.1
    rl_emb = np.zeros(10, dtype=np.float32)
    # Occupancy placeholders scaled from queue
    occ = np.array([
        min(queue_len, 5) / 5.0,
        min(max(queue_len - 5, 0), 5) / 5.0,
        0.2,
        0.2,
    ], dtype=np.float32)
    time_feats = np.array([
        datetime.utcnow().hour / 24.0,
        datetime.utcnow().minute / 60.0,
    ], dtype=np.float32)
    state = np.concatenate([
        probs.astype(np.float32),
        np.array([risk], dtype=np.float32),
        rl_emb,
        occ,
        time_feats,
    ])
    return state


def get_operational_action(rl_model, ed_state: np.ndarray) -> str:
    if rl_model is None:
        return ACTION_MAP.get(1, "Move to Waiting Room")
    action_idx, _ = rl_model.predict(ed_state, deterministic=True)
    action_idx = int(action_idx)
    return ACTION_MAP.get(action_idx, "Move to Waiting Room")

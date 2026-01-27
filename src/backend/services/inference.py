from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import shap
from typing import Dict, Any, Optional


# Median values from NHAMCS dataset for imputation
# These are used when vitals are missing or invalid
MEDIAN_VITALS = {
    "Age": 45.0,
    "Pulse": 82.0,
    "HR": 82.0,
    "SBP": 130.0,
    "DBP": 78.0,
    "Resp": 18.0,
    "RR": 18.0,
    "O2Sat": 98.0,
    "Temp": 98.2,
    "PainScale": 5.0,
}

# Valid ranges for vital signs (used for outlier detection)
VALID_RANGES = {
    "Age": (0, 120),
    "Pulse": (20, 250),
    "HR": (20, 250),
    "SBP": (40, 300),
    "DBP": (20, 200),
    "Resp": (4, 60),
    "RR": (4, 60),
    "O2Sat": (50, 100),
    "Temp": (86.0, 113.0),  # Fahrenheit
    "PainScale": (0, 10),
}

# Required columns for the preprocessor (in exact order)
REQUIRED_COLUMNS = ["Age", "Temp", "Pulse", "Resp", "SBP", "DBP", "O2Sat", "PainScale", "ArrivalMode"]


def sanitize_vitals(vitals: Dict[str, Any]) -> Dict[str, float]:
    """
    Sanitize and impute missing/invalid vital signs.
    
    - Missing values → median imputation
    - Out-of-range values → clamped to valid range
    - Negative values → treated as missing
    """
    sanitized = {}
    
    for key, default in MEDIAN_VITALS.items():
        value = vitals.get(key)
        
        # Handle missing or invalid types
        if value is None or not isinstance(value, (int, float)):
            sanitized[key] = default
            continue
        
        # Handle negative values as missing
        if value < 0:
            sanitized[key] = default
            continue
        
        # Handle zero values - treat as missing for most vitals
        if value == 0 and key in ["Pulse", "HR", "SBP", "DBP", "Resp", "RR", "O2Sat"]:
            sanitized[key] = default
            continue
        
        # Clamp to valid range if specified
        if key in VALID_RANGES:
            min_val, max_val = VALID_RANGES[key]
            value = max(min_val, min(max_val, float(value)))
        
        sanitized[key] = float(value)
    
    # Copy through any other fields (like patient_id)
    for key, value in vitals.items():
        if key not in sanitized:
            sanitized[key] = value
    
    return sanitized


def get_bert_embedding(text: str, tokenizer, model, device: torch.device) -> np.ndarray:
    # Handle empty or very short text
    if not text or len(text.strip()) < 3:
        text = "General complaint"
    
    # Truncate extremely long text
    if len(text) > 5000:
        text = text[:5000]
    
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
    return embeddings.flatten()


def prepare_structured_features(payload_static: Dict[str, float], arrival_mode: Optional[str]) -> pd.DataFrame:
    """Prepare structured features with proper imputation for missing values."""
    # First sanitize the vitals
    sanitized = sanitize_vitals(payload_static)
    
    # Handle arrival mode
    arrival_map = {"Ambulance": 1.0, "Public Transport": 2.0, "Walk-in": 3.0, "Other": 4.0}
    arrival_val = 3.0  # Default to Walk-in
    if arrival_mode:
        arrival_val = arrival_map.get(arrival_mode, 3.0)
    elif "ArrivalMode" in sanitized:
        arrival_val = float(sanitized.get("ArrivalMode", 3.0))
    
    # Build DataFrame with EXACTLY the columns the preprocessor expects
    # Map common aliases to expected column names
    df_data = {
        "Age": sanitized.get("Age", MEDIAN_VITALS["Age"]),
        "Temp": sanitized.get("Temp", MEDIAN_VITALS["Temp"]),
        "Pulse": sanitized.get("Pulse") or sanitized.get("HR", MEDIAN_VITALS["Pulse"]),
        "Resp": sanitized.get("Resp") or sanitized.get("RR", MEDIAN_VITALS["Resp"]),
        "SBP": sanitized.get("SBP", MEDIAN_VITALS["SBP"]),
        "DBP": sanitized.get("DBP", MEDIAN_VITALS["DBP"]),
        "O2Sat": sanitized.get("O2Sat", MEDIAN_VITALS["O2Sat"]),
        "PainScale": sanitized.get("PainScale", MEDIAN_VITALS["PainScale"]),
        "ArrivalMode": arrival_val,
    }
    
    # Create DataFrame with columns in the exact order expected
    df = pd.DataFrame({col: [df_data[col]] for col in REQUIRED_COLUMNS})
    
    return df


def combine_features(preprocessor, structured_df: pd.DataFrame, embedding: np.ndarray) -> np.ndarray:
    X_struct = preprocessor.transform(structured_df)
    if hasattr(X_struct, "toarray"):
        X_struct = X_struct.toarray()
    return np.hstack([X_struct, embedding.reshape(1, -1)])


def predict_sup(model, X_combined: np.ndarray) -> tuple[int, float, np.ndarray]:
    probs = model.predict_proba(X_combined)[0]
    pred_idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    return pred_idx + 1, conf, probs  # convert to 1-based level


def compute_shap_values(model, X_combined: np.ndarray):
    # Expecting an sklearn-compatible model (e.g., LGBM inside stacking pipeline)
    try:
        lgbm = model.named_steps["classifier"].estimators_[0]
    except Exception:
        lgbm = None
    if lgbm is None:
        return None
    explainer = shap.TreeExplainer(lgbm)
    shap_vals = explainer.shap_values(X_combined)
    return shap_vals

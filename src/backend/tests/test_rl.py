"""RL Observation Space Validation Tests.

Ensures the build_ed_state vector matches the DQN agent's expected input.
"""

import os
import warnings

import numpy as np
import pytest
from stable_baselines3 import DQN

# Adjust path for imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services.priority import build_ed_state, ACTION_MAP

MODEL_PATH = "data/dqn_triage_agent.zip"

# Expected observation space dimensions based on training:
# probs(5) + risk(1) + rl_emb(10) + occ(4) + time(2) = 22
EXPECTED_OBS_DIM = 22


@pytest.fixture
def rl_model():
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"RL model not found at {MODEL_PATH}")
    return DQN.load(MODEL_PATH)


class MockDB:
    """Mock database session for testing without real DB."""

    def query(self, *args, **kwargs):
        return self

    def filter(self, *args, **kwargs):
        return self

    def count(self):
        return 3  # Mock 3 waiting patients

    def with_entities(self, *args, **kwargs):
        return self

    def scalar(self):
        return 15.5  # Mock 15.5 minutes avg wait


def test_observation_shape_matches_model(rl_model):
    """Verify generated observation matches model's expected input shape."""
    expected_shape = rl_model.observation_space.shape
    assert len(expected_shape) == 1, "Expected 1D observation space"

    mock_probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
    mock_vitals = {"O2Sat": 95, "SBP": 120, "HR": 80, "Temp": 98.6}
    mock_db = MockDB()

    ed_state = build_ed_state(mock_probs, mock_vitals, mock_db)

    assert ed_state.shape == expected_shape, (
        f"Observation shape mismatch: got {ed_state.shape}, expected {expected_shape}"
    )


def test_observation_dimension():
    """Test observation vector has correct dimension without loading model."""
    mock_probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
    mock_vitals = {"O2Sat": 95, "SBP": 120}
    mock_db = MockDB()

    ed_state = build_ed_state(mock_probs, mock_vitals, mock_db)

    assert ed_state.shape == (EXPECTED_OBS_DIM,), (
        f"Expected {EXPECTED_OBS_DIM} dims, got {ed_state.shape[0]}"
    )


def test_observation_dtype():
    """Ensure observation is float32 as required by stable-baselines3."""
    mock_probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
    mock_vitals = {"O2Sat": 95, "SBP": 120}
    mock_db = MockDB()

    ed_state = build_ed_state(mock_probs, mock_vitals, mock_db)

    assert ed_state.dtype == np.float32, f"Expected float32, got {ed_state.dtype}"


def test_observation_bounds(rl_model):
    """Check observation values fall within model's expected range."""
    obs_space = rl_model.observation_space
    low = obs_space.low
    high = obs_space.high

    mock_probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
    mock_vitals = {"O2Sat": 95, "SBP": 120}
    mock_db = MockDB()

    ed_state = build_ed_state(mock_probs, mock_vitals, mock_db)

    out_of_bounds = []
    for i, (val, lo, hi) in enumerate(zip(ed_state, low, high)):
        if val < lo or val > hi:
            out_of_bounds.append(f"Feature {i}: {val} not in [{lo}, {hi}]")

    if out_of_bounds:
        warnings.warn(
            f"Observation values out of training bounds:\n" + "\n".join(out_of_bounds)
        )


def test_risk_feature_normalization():
    """Verify risk feature is properly normalized (0.1 or 0.9)."""
    mock_probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
    mock_db = MockDB()

    # Normal vitals -> low risk
    normal_vitals = {"O2Sat": 98, "SBP": 120}
    ed_state_normal = build_ed_state(mock_probs, normal_vitals, mock_db)
    risk_idx = 5  # probs(5) + risk at index 5
    assert ed_state_normal[risk_idx] == pytest.approx(0.1), "Expected low risk 0.1"

    # Critical vitals -> high risk
    critical_vitals = {"O2Sat": 85, "SBP": 80}
    ed_state_critical = build_ed_state(mock_probs, critical_vitals, mock_db)
    assert ed_state_critical[risk_idx] == pytest.approx(0.9), "Expected high risk 0.9"


def test_time_features_normalized():
    """Ensure time features are normalized to [0, 1]."""
    mock_probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
    mock_vitals = {"O2Sat": 95, "SBP": 120}
    mock_db = MockDB()

    ed_state = build_ed_state(mock_probs, mock_vitals, mock_db)

    # Time features are last 2 elements
    hour_feat = ed_state[-2]
    minute_feat = ed_state[-1]

    assert 0 <= hour_feat <= 1, f"Hour feature {hour_feat} not normalized"
    assert 0 <= minute_feat <= 1, f"Minute feature {minute_feat} not normalized"


def test_action_prediction_valid(rl_model):
    """Verify RL model returns valid action indices."""
    mock_probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
    mock_vitals = {"O2Sat": 95, "SBP": 120}
    mock_db = MockDB()

    ed_state = build_ed_state(mock_probs, mock_vitals, mock_db)
    action, _ = rl_model.predict(ed_state, deterministic=True)

    assert int(action) in ACTION_MAP, f"Invalid action {action}, expected one of {list(ACTION_MAP.keys())}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

import os
from contextlib import asynccontextmanager

import numpy as np
import joblib
import torch
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from .config import settings
from .database import SessionLocal, init_db, DATABASE_URL
from .routers import auth, triage, dashboard


MODEL_PATHS = {
    "sup_model": "data/nhamcs_bert_model.joblib",
    "preprocessor": "data/nhamcs_preprocessor.joblib",
    "lstm_model": "data/lstm_model.pt",
    "rl_model": "data/dqn_triage_agent.zip",
}

# Development mode: use mock models when real ones unavailable
# Auto-detect based on model file existence
DEV_MODE = os.environ.get("TRIAGE_DEV_MODE", "").lower() == "true" or not os.path.exists(MODEL_PATHS["sup_model"])


class MockSupervisedModel:
    """Mock supervised model for development/testing."""
    
    def predict_proba(self, X):
        # Simulate KTAS prediction based on feature values
        # Higher values in first few features → higher acuity
        n_samples = X.shape[0]
        probs = np.random.dirichlet(np.ones(5), size=n_samples)
        # Bias toward middle acuity
        probs[:, 2] += 0.2
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs


class MockPreprocessor:
    """Mock preprocessor for development/testing."""
    
    def transform(self, df):
        # Return fixed-size feature vector
        return np.random.randn(len(df), 20)


class MockRLModel:
    """Mock RL model for development/testing."""
    
    def predict(self, observation, deterministic=True):
        # Randomly pick an action
        return np.random.randint(0, 4), None


def _ensure_exists(path: str, name: str):
    if not os.path.exists(path):
        raise RuntimeError(f"{name} not found at {path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if we should use mock models (dev mode or missing files)
    use_mocks = DEV_MODE or not os.path.exists(MODEL_PATHS["sup_model"])
    
    if use_mocks:
        print("⚠️  DEV MODE: Using mock models for development/testing")
        app.state.sup_model = MockSupervisedModel()
        app.state.preprocessor = MockPreprocessor()
        app.state.rl_model = MockRLModel()
        app.state.tokenizer = None
        app.state.bert_model = None
        app.state.lstm_model = None
        app.state.device = device
        app.state.dev_mode = True
    else:
        # Production mode: load real models
        from transformers import AutoModel, AutoTokenizer
        from stable_baselines3 import DQN
        
        _ensure_exists(MODEL_PATHS["sup_model"], "Supervised model")
        _ensure_exists(MODEL_PATHS["preprocessor"], "Preprocessor")

        sup_model = joblib.load(MODEL_PATHS["sup_model"])
        preprocessor = joblib.load(MODEL_PATHS["preprocessor"])

        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        bert_model.to(device)
        bert_model.eval()

        lstm_model = None
        if os.path.exists(MODEL_PATHS["lstm_model"]):
            lstm_model = torch.load(MODEL_PATHS["lstm_model"], map_location=device)
            if hasattr(lstm_model, "eval"):
                lstm_model.eval()

        rl_model = None
        if os.path.exists(MODEL_PATHS["rl_model"]):
            rl_model = DQN.load(MODEL_PATHS["rl_model"], device=device)

        app.state.device = device
        app.state.sup_model = sup_model
        app.state.preprocessor = preprocessor
        app.state.tokenizer = tokenizer
        app.state.bert_model = bert_model
        app.state.lstm_model = lstm_model
        app.state.rl_model = rl_model
        app.state.dev_mode = False
    
    # Initialize database (creates tables for SQLite)
    if DATABASE_URL.startswith("sqlite"):
        print(f"📦 Initializing SQLite database: {DATABASE_URL}")
        init_db()

    try:
        yield
    finally:
        for attr in ["sup_model", "preprocessor", "tokenizer", "bert_model", "lstm_model", "rl_model", "device"]:
            if hasattr(app.state, attr):
                delattr(app.state, attr)


app = FastAPI(title=settings.app_name, lifespan=lifespan)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(triage.router)
app.include_router(dashboard.router)


def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health_check(db: Session = Depends(get_db_session)):
    db_status = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_status = "unavailable"
    
    models_loaded = hasattr(app.state, "sup_model") and app.state.sup_model is not None
    dev_mode = getattr(app.state, "dev_mode", False)
    
    return {
        "status": "ok" if (db_status == "ok" or dev_mode) else "degraded",
        "database": db_status,
        "models": "mock" if dev_mode else ("loaded" if models_loaded else "not_loaded"),
        "dev_mode": dev_mode,
    }

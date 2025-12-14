import os
import joblib
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from stable_baselines3 import DQN
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def check_file(path):
    if os.path.exists(path):
        print(f"[OK] Found {path}")
        return True
    else:
        print(f"[FAIL] Missing {path}")
        return False

def verify_system():
    print("--- System Verification Started ---")
    
    # 1. Check Critical Files
    files_to_check = [
        'data/nhamcs_combined.csv',
        'data/nhamcs_bert_features.npy',
        'data/nhamcs_bert_model.joblib',
        'data/dqn_triage_agent.zip',
        'data/rl_ready_data_nhamcs.csv',
        'output/hybrid_evaluation_results.csv'
    ]
    
    all_files_ok = all([check_file(f) for f in files_to_check])
    if not all_files_ok:
        print("CRITICAL: Some files are missing. Please run the pipeline scripts.")
        return

    # 2. Load Models
    print("\n--- Loading Models ---")
    try:
        print("Loading Supervised Model...")
        sup_model = joblib.load('data/nhamcs_bert_model.joblib')
        print("[OK] Supervised Model Loaded")
        
        print("Loading RL Agent...")
        rl_model = DQN.load("data/dqn_triage_agent.zip")
        print("[OK] RL Agent Loaded")
        
        print("Loading BERT...")
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        print("[OK] BERT Loaded")
        
    except Exception as e:
        print(f"[FAIL] Model Loading Error: {e}")
        return

    # 3. Simulate App Prediction Logic
    print("\n--- Simulating Prediction Pipeline ---")
    try:
        # Mock Input
        input_data = pd.DataFrame({
            'Age': [45],
            'Temp': [98.6],
            'Pulse': [80],
            'Resp': [16],
            'SBP': [120],
            'DBP': [80],
            'O2Sat': [98],
            'PainScale': [5],
            'ArrivalMode': ['Walk-in']
        })
        
        # Mock BERT Embedding
        text = "Chest pain"
        inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = bert_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
            
        # Preprocessing (Replicating App Logic)
        # We need to fit the preprocessor on training data first
        print("Fitting Preprocessor on sample data...")
        train_df = pd.read_csv('data/nhamcs_combined.csv', nrows=100)
        num_cols = ['Age', 'Temp', 'Pulse', 'Resp', 'SBP', 'DBP', 'O2Sat', 'PainScale']
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('cat', categorical_transformer, ['ArrivalMode'])
            ])
        
        preprocessor.fit(train_df[num_cols + ['ArrivalMode']])
        X_struct = preprocessor.transform(input_data)
        if hasattr(X_struct, "toarray"): X_struct = X_struct.toarray()
        
        # Combine
        X_combined = np.hstack([X_struct, embedding.reshape(1, -1)])
        print(f"Combined Input Shape: {X_combined.shape}")
        
        # Predict Supervised
        probs = sup_model.predict_proba(X_combined)[0]
        print(f"[OK] Supervised Prediction: {probs}")
        
        # Predict RL
        # [Probs(5), Risk(1), Embedding(10), Occ(4), Time(2)]
        # Note: RL Agent expects 10-dim embedding (from old pipeline) or 768?
        # Let's check what we trained it with.
        # In `run_hybrid_inference.py`, we used `embedding = np.zeros(10)`.
        # So we MUST use zeros(10) here.
        
        rl_emb = np.zeros(10)
        occ = [0.1, 0.2, 0.3, 0.2]
        time = [0.5, 0.5]
        risk = 0.1
        
        obs = np.concatenate([probs, [risk], rl_emb, occ, time]).astype(np.float32)
        action, _ = rl_model.predict(obs, deterministic=True)
        print(f"[OK] RL Action: {action}")
        
        print("\n[SUCCESS] System Verification Complete. All components are functional.")
        
    except Exception as e:
        print(f"\n[FAIL] Prediction Pipeline Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_system()

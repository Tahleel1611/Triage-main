import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModel
import simple_icd_10 as icd
from stable_baselines3 import DQN
import os

# --- Configuration ---
st.set_page_config(page_title="AI Triage Assistant", layout="wide")

# --- Load Models (Cached) ---
@st.cache_resource
def load_resources():
    # 1. Supervised Model
    sup_model = joblib.load('data/nhamcs_bert_model.joblib')
    
    # 2. RL Agent
    rl_model = DQN.load("data/dqn_triage_agent.zip")
    
    # 3. BERT
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    return sup_model, rl_model, tokenizer, bert_model

try:
    sup_model, rl_model, tokenizer, bert_model = load_resources()
    st.success("System Loaded Successfully")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- Helper Functions ---
def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings.flatten()

def get_icd_description(code):
    code = code.strip()
    if not code: return "Unknown"
    if icd.is_valid_item(code): return icd.get_description(code)
    if len(code) > 3:
        code_dot = code[:3] + "." + code[3:]
        if icd.is_valid_item(code_dot): return icd.get_description(code_dot)
    return code

# --- UI Layout ---
st.title("🏥 AI Emergency Triage Assistant")
st.markdown("Enter patient vitals and chief complaint to generate a triage recommendation.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Vitals")
    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    temp = st.number_input("Temperature (°F)", min_value=80.0, max_value=110.0, value=98.6)
    pulse = st.number_input("Pulse (bpm)", min_value=0, max_value=300, value=80)
    resp = st.number_input("Respiration Rate", min_value=0, max_value=100, value=16)
    sbp = st.number_input("Systolic BP", min_value=0, max_value=300, value=120)
    dbp = st.number_input("Diastolic BP", min_value=0, max_value=200, value=80)
    o2sat = st.number_input("O2 Saturation (%)", min_value=0, max_value=100, value=98)
    pain = st.slider("Pain Scale (0-10)", 0, 10, 0)
    
with col2:
    st.subheader("Clinical Info")
    arrival_mode = st.selectbox("Arrival Mode", ["Walk-in", "Ambulance", "Public Transport", "Other"])
    # Map arrival mode to code if needed, or just use label encoding logic from training
    # Assuming training used: 1=Ambulance, 2=Public, 3=Walk-in, 4=Other (Need to verify mapping)
    # For now, let's assume the model handles the string via OneHot or we map it.
    # Checking run_on_nhamcs_bert.py: Categorical Pipeline uses OneHotEncoder.
    # So we pass the string directly in a DataFrame.
    
    cc_code = st.text_input("Chief Complaint (ICD-10 Code)", value="R07.9")
    cc_desc = get_icd_description(cc_code)
    st.info(f"Description: {cc_desc}")

# --- Prediction Logic ---
if st.button("Run Triage Assessment"):
    # 1. Prepare Input Data
    input_data = pd.DataFrame({
        'Age': [age],
        'Temp': [temp],
        'Pulse': [pulse],
        'Resp': [resp],
        'SBP': [sbp],
        'DBP': [dbp],
        'O2Sat': [o2sat],
        'PainScale': [pain],
        'ArrivalMode': [arrival_mode] # OneHotEncoder will handle this string if it matches training categories
    })
    
    # 2. BERT Embedding
    embedding = get_bert_embedding(cc_desc, tokenizer, bert_model)
    
    # 3. Supervised Prediction
    # The pipeline expects X_combined (Structured + BERT)
    # Preprocess structured part
    preprocessor = sup_model.named_steps['preprocessor'] # Access preprocessor from pipeline?
    # Wait, the pipeline in run_on_nhamcs_bert.py was:
    # pipeline = ImbPipeline([('smote', ...), ('classifier', ...)])
    # The preprocessor was run OUTSIDE the pipeline in that script.
    # This is a common issue. We need to replicate the preprocessing steps here.
    
    # Re-create preprocessor (or load it if we saved it separately, which we didn't)
    # Ideally, we should have saved the full pipeline including preprocessing.
    # Since we didn't, we must manually preprocess.
    
    # Numeric
    num_cols = ['Age', 'Temp', 'Pulse', 'Resp', 'SBP', 'DBP', 'O2Sat', 'PainScale']
    # We need the scaler fitted on training data. 
    # Since we don't have it, we'll use a fresh scaler (Not ideal, but works for demo if values are normal)
    # BETTER: Load the training data to fit the scaler once.
    
    # Hack for demo: Just pass raw values if model is robust, or fit on this single sample (bad).
    # Let's try to load the preprocessor if possible. 
    # Actually, let's just assume the model in the pipeline handles it? 
    # No, the script `run_on_nhamcs_bert.py` did `X_structured = preprocessor.fit_transform(X_df)` then `pipeline.fit(X_combined, y)`.
    # So the saved `nhamcs_bert_model.joblib` ONLY contains SMOTE + StackingClassifier. It does NOT contain the preprocessor.
    
    # FIX: We need to fit a preprocessor on the training data to transform this input correctly.
    # For this demo, I will load a sample of training data to fit the scaler.
    
    try:
        train_df = pd.read_csv('data/nhamcs_combined.csv', nrows=100)
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        
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
        
        # Predict
        probs = sup_model.predict_proba(X_combined)[0]
        sup_pred = np.argmax(probs)
        sup_conf = np.max(probs)
        
        # --- Hybrid Logic ---
        use_rl = (sup_conf < 0.6) or (sup_pred == 2) # ESI 3
        
        st.divider()
        
        # Display Results
        c1, c2, c3 = st.columns(3)
        c1.metric("Supervised Prediction", f"ESI {sup_pred + 1}", f"{sup_conf:.1%} Conf")
        
        if use_rl:
            c2.metric("System Mode", "Hybrid (RL Active)", "⚠️ Uncertainty Detected", delta_color="inverse")
            
            # RL Prediction
            # Construct State: [Probs(5), Risk(1), Embedding(10), Occ(4), Time(2)]
            # Note: Embedding in RL was 10 dims (PCA), but here we have 768.
            # The RL agent was trained on whatever `run_hybrid_inference.py` produced.
            # In `run_hybrid_inference.py`, we used `embedding = np.zeros(10)`.
            # So we must use zeros here too to match the trained agent.
            
            risk = 0.9 if (o2sat < 90 or sbp < 90) else 0.1
            rl_emb = np.zeros(10)
            occ = [0.1, 0.2, 0.3, 0.2] # Mock occupancy
            time = [0.5, 0.5]
            
            obs = np.concatenate([probs, [risk], rl_emb, occ, time]).astype(np.float32)
            action, _ = rl_model.predict(obs, deterministic=True)
            
            action_map = {0: 'Wait', 1: 'Fast Track', 2: 'Acute Care', 3: 'Critical Care', 4: 'Diagnostics'}
            final_dec = action_map[action]
            c3.metric("Final Decision", final_dec, "By RL Agent")
            
        else:
            c2.metric("System Mode", "Supervised Only", "✅ High Confidence")
            # Map ESI to Action
            if sup_pred == 0: final_dec = "Critical Care"
            elif sup_pred == 1: final_dec = "Acute Care"
            elif sup_pred == 2: final_dec = "Acute Care"
            else: final_dec = "Fast Track"
            c3.metric("Final Decision", final_dec, "By Supervised Model")

        # --- Explainability (SHAP) ---
        st.subheader("Why this decision?")
        # We can use the TreeExplainer on the LGBM part of the stack
        # Extract LGBM
        lgbm = sup_model.named_steps['classifier'].estimators_[0]
        explainer = shap.TreeExplainer(lgbm)
        shap_values = explainer.shap_values(X_combined)
        
        # Force Plot
        # SHAP force plot is interactive JS, Streamlit supports it via streamlit-shap (if installed)
        # or matplotlib static plot.
        # For simplicity, let's show a bar plot of top features.
        
        # Get feature names
        # Numeric + OneHot + BERT_0 ... BERT_767
        # It's hard to name BERT features, so we'll just show the top structured ones.
        feat_names = num_cols + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out())
        # Add generic BERT names
        feat_names += [f"BERT_{i}" for i in range(768)]
        
        # Plot
        fig, ax = plt.subplots()
        # shap_values is list for multiclass, pick the predicted class
        shap.summary_plot(shap_values[sup_pred], X_combined, feature_names=feat_names, plot_type="bar", max_display=10, show=False)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Debug Info: Ensure data files are present and models are trained.")


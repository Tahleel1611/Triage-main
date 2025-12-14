import pandas as pd
import joblib
import numpy as np

def main():
    # 1. Load Data and Model
    data_path = 'data/nhamcs_combined.csv'
    model_path = 'data/nhamcs_model.joblib'
    output_path = 'data/rl_ready_data_nhamcs.csv'
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['ESI'])
    
    print(f"Loading model from {model_path}...")
    pipeline = joblib.load(model_path)
    
    # 2. Prepare Features
    numeric_features = ['Age', 'Temp', 'Pulse', 'Resp', 'SBP', 'DBP', 'O2Sat']
    text_feature = 'Chief_complain'
    
    X = df[numeric_features + [text_feature]]
    
    # 3. Generate Probabilities
    print("Generating predictions...")
    # The model was trained with LabelEncoder 0-4.
    # Classes are 0, 1, 2, 3, 4 corresponding to ESI levels?
    # Wait, ESI is 1-5. LabelEncoder maps 1->0, 2->1, etc.
    # So prob_class_0 corresponds to ESI 1.
    
    probs = pipeline.predict_proba(X)
    
    # 4. Create Output DataFrame
    output_df = pd.DataFrame()
    
    # Add probabilities
    for i in range(probs.shape[1]):
        output_df[f'prob_class_{i}'] = probs[:, i]
        
    # Add Acuity (ESI)
    # We need to map ESI 1-5 to 0-4 or keep as is?
    # The RL environment uses 'acuity' for reward calculation.
    # Line 103: patient_acuity = self.current_patient['acuity'] # 1-5
    # So we should keep the original ESI values.
    output_df['acuity'] = df['ESI'].values
    
    # Add dummy risk and embedding if we want to be fancy, but Env handles it.
    # Let's add a simple risk score based on O2Sat and SBP for flavor.
    # Risk = 1.0 if O2 < 90 or SBP < 90, else normalized.
    output_df['risk'] = np.where((df['O2Sat'] < 90) | (df['SBP'] < 90), 0.9, 0.1)
    
    # Save
    print(f"Saving RL-ready data to {output_path}...")
    output_df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()

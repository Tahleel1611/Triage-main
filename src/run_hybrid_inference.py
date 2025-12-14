import pandas as pd
import numpy as np
from stable_baselines3 import DQN
import os

def main():
    # 1. Load Data
    data_path = "data/rl_ready_data_nhamcs.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Load RL Model
    model_path = "data/dqn_triage_agent.zip"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
        
    print(f"Loading RL agent from {model_path}...")
    rl_model = DQN.load(model_path)
    
    # 3. Hybrid Inference Loop
    print("Running Hybrid Inference...")
    
    critical_misses = 0
    total_critical = 0
    
    over_triage = 0
    total_non_critical = 0
    
    correct_triage = 0
    
    rl_intervention_count = 0
    
    # Mock ED State
    ed_state = {
        'waiting': 5,
        'critical_beds': 2,
        'acute_beds': 10,
        'fast_track': 5
    }
    
    results = []
    
    for idx, row in df.iterrows():
        # --- 1. Supervised Model Output ---
        # Extract probs
        prob_cols = [c for c in df.columns if 'prob_class_' in c]
        probs = row[prob_cols].values.astype(float)
        if len(probs) < 5:
            probs = np.pad(probs, (0, 5 - len(probs)))
            
        sup_pred_idx = np.argmax(probs) # 0=ESI 1, 1=ESI 2, ...
        sup_confidence = np.max(probs)
        
        # --- 2. Hybrid Logic ---
        # Condition: Uncertain (< 0.6) OR Predicted ESI 3 (Idx 2)
        use_rl = (sup_confidence < 0.6) or (sup_pred_idx == 2)
        
        final_action = 0
        source = "Supervised"
        
        if use_rl:
            source = "RL_Agent"
            rl_intervention_count += 1
            
            # Construct RL Observation
            risk = row['risk'] if 'risk' in row else 0.5
            embedding = np.zeros(10)
            
            obs = np.concatenate([
                probs,
                [risk],
                embedding,
                list(ed_state.values()),
                [30.0],
                [0.5, 0.5]
            ]).astype(np.float32)
            
            # Predict
            action, _ = rl_model.predict(obs, deterministic=True)
            final_action = action
            
        else:
            # Map Supervised ESI to Action
            # ESI 1 (0) -> Critical (3)
            # ESI 2 (1) -> Acute (2)
            # ESI 3 (2) -> Acute (2) (Though logic says we use RL for ESI 3, keeping mapping for completeness)
            # ESI 4 (3) -> Fast Track (1)
            # ESI 5 (4) -> Fast Track (1)
            
            if sup_pred_idx == 0: final_action = 3
            elif sup_pred_idx == 1: final_action = 2
            elif sup_pred_idx == 2: final_action = 2
            elif sup_pred_idx == 3: final_action = 1
            elif sup_pred_idx == 4: final_action = 1
            
        # --- 3. Evaluation ---
        acuity = row['acuity'] # Actual ESI 1-5
        
        is_critical_patient = (acuity <= 2) # ESI 1, 2
        is_low_acuity_patient = (acuity >= 4) # ESI 4, 5
        
        # Critical Miss
        if is_critical_patient:
            total_critical += 1
            if final_action == 0 or final_action == 1:
                critical_misses += 1
                
        # Over Triage
        if is_low_acuity_patient:
            total_non_critical += 1
            if final_action == 2 or final_action == 3:
                over_triage += 1
                
        # Correctness
        if acuity <= 2 and final_action in [2, 3]:
            correct_triage += 1
        elif acuity == 3 and final_action == 2:
            correct_triage += 1
        elif acuity >= 4 and final_action == 1:
            correct_triage += 1
            
        results.append({
            'ESI': acuity,
            'Sup_Pred': sup_pred_idx + 1,
            'Sup_Conf': sup_confidence,
            'Source': source,
            'Final_Action': final_action,
            'Is_Critical_Miss': (is_critical_patient and final_action in [0, 1])
        })
            
    # 4. Report
    print("\n--- Hybrid System Evaluation Report ---")
    print(f"Total Patients: {len(df)}")
    print(f"RL Agent Interventions: {rl_intervention_count} ({rl_intervention_count/len(df)*100:.1f}%)")
    
    cm_rate = (critical_misses / total_critical * 100) if total_critical > 0 else 0
    ot_rate = (over_triage / total_non_critical * 100) if total_non_critical > 0 else 0
    acc_rate = (correct_triage / len(df) * 100)
    
    print(f"\nCritical Patients (ESI 1 & 2): {total_critical}")
    print(f"Critical Misses: {critical_misses}")
    print(f"Critical Miss Rate: {cm_rate:.2f}%")
    
    print(f"\nLow Acuity Patients (ESI 4 & 5): {total_non_critical}")
    print(f"Over-Triage Count: {over_triage}")
    print(f"Over-Triage Rate: {ot_rate:.2f}%")
    
    print(f"\nOverall 'Correct' Allocation Rate: {acc_rate:.2f}%")
    
    # Save
    res_df = pd.DataFrame(results)
    res_df.to_csv("output/hybrid_evaluation_results.csv", index=False)
    print("\nDetailed results saved to output/hybrid_evaluation_results.csv")
    
    with open("output/hybrid_evaluation_report.txt", "w") as f:
        f.write("Hybrid System Evaluation Report\n")
        f.write("===============================\n")
        f.write(f"RL Interventions: {rl_intervention_count/len(df)*100:.1f}%\n")
        f.write(f"Critical Miss Rate: {cm_rate:.2f}%\n")
        f.write(f"Over-Triage Rate: {ot_rate:.2f}%\n")
        f.write(f"Overall Accuracy: {acc_rate:.2f}%\n")

if __name__ == "__main__":
    main()

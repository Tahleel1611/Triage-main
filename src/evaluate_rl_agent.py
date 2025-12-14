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
    
    # 2. Load Model
    model_path = "data/dqn_triage_agent.zip"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
        
    print(f"Loading model from {model_path}...")
    model = DQN.load(model_path)
    
    # 3. Evaluation Loop
    print("Evaluating agent on all patients...")
    
    critical_misses = 0
    total_critical = 0
    
    over_triage = 0
    total_non_critical = 0
    
    correct_triage = 0
    
    # Action Map
    # 0: Waiting Room
    # 1: Fast Track
    # 2: Acute Care Bed
    # 3: Critical Care
    # 4: Order Diagnostics
    
    # Mock ED State (Half Full) to test pure decision making
    # We want to see if the agent recognizes severity, assuming it HAS resources.
    ed_state = {
        'waiting': 5,
        'critical_beds': 2, # Some available
        'acute_beds': 10,   # Some available
        'fast_track': 5     # Some available
    }
    
    results = []
    
    for idx, row in df.iterrows():
        # Construct Observation
        # Must match rl_environment.py _get_observation structure
        
        # Extract probs
        prob_cols = [c for c in df.columns if 'prob_class_' in c]
        probs = row[prob_cols].values.astype(float)
        if len(probs) < 5:
            probs = np.pad(probs, (0, 5 - len(probs)))
            
        risk = row['risk'] if 'risk' in row else 0.5
        embedding = np.zeros(10) # Placeholder as used in env
        
        obs = np.concatenate([
            probs,
            [risk],
            embedding,
            list(ed_state.values()),
            [30.0], # Mock wait time
            [0.5, 0.5] # Mock time
        ]).astype(np.float32)
        
        # Predict
        action, _ = model.predict(obs, deterministic=True)
        
        # Evaluate
        acuity = row['acuity'] # 1-5
        
        # Metrics
        is_critical_patient = (acuity <= 2) # ESI 1, 2
        is_low_acuity_patient = (acuity >= 4) # ESI 4, 5
        
        # Critical Miss: Critical Patient sent to Wait (0) or Fast Track (1)
        if is_critical_patient:
            total_critical += 1
            if action == 0 or action == 1:
                critical_misses += 1
                
        # Over Triage: Low Acuity sent to Critical (3) or Acute (2)
        if is_low_acuity_patient:
            total_non_critical += 1
            if action == 2 or action == 3:
                over_triage += 1
                
        # General "Correctness" (Loose definition)
        # ESI 1/2 -> Critical/Acute (3/2)
        # ESI 3 -> Acute (2)
        # ESI 4/5 -> Fast Track (1)
        # Wait (0) is valid if beds full, but here beds are available.
        if acuity <= 2 and action in [2, 3]:
            correct_triage += 1
        elif acuity == 3 and action == 2:
            correct_triage += 1
        elif acuity >= 4 and action == 1:
            correct_triage += 1
            
        results.append({
            'ESI': acuity,
            'Action': action,
            'Is_Critical_Miss': (is_critical_patient and action in [0, 1])
        })
            
    # 4. Report
    print("\n--- RL Agent Evaluation Report ---")
    print(f"Total Patients: {len(df)}")
    
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
    
    # Save detailed results
    res_df = pd.DataFrame(results)
    res_df.to_csv("output/rl_evaluation_results.csv", index=False)
    print("\nDetailed results saved to output/rl_evaluation_results.csv")
    
    with open("output/rl_evaluation_report.txt", "w") as f:
        f.write("RL Agent Evaluation Report\n")
        f.write("==========================\n")
        f.write(f"Critical Miss Rate: {cm_rate:.2f}%\n")
        f.write(f"Over-Triage Rate: {ot_rate:.2f}%\n")
        f.write(f"Overall Accuracy: {acc_rate:.2f}%\n")

if __name__ == "__main__":
    main()

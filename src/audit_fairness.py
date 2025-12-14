import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # 1. Load Data
    original_data_path = 'data/nhamcs_combined.csv'
    results_path = 'output/hybrid_evaluation_results.csv'
    
    if not os.path.exists(original_data_path) or not os.path.exists(results_path):
        print("Error: Data files not found.")
        return

    print("Loading data...")
    df_orig = pd.read_csv(original_data_path)
    df_results = pd.read_csv(results_path)
    
    # 2. Align Data
    # The results were generated from df_orig after dropping rows with missing ESI.
    df_orig = df_orig.dropna(subset=['ESI'])
    
    # Reset index to ensure alignment
    df_orig = df_orig.reset_index(drop=True)
    df_results = df_results.reset_index(drop=True)
    
    if len(df_orig) != len(df_results):
        print(f"Warning: Length mismatch! Original: {len(df_orig)}, Results: {len(df_results)}")
        # Truncate to shorter length (likely results are shorter if something else was dropped)
        min_len = min(len(df_orig), len(df_results))
        df_orig = df_orig.iloc[:min_len]
        df_results = df_results.iloc[:min_len]
        
    # Merge
    # df_results likely has 'ESI' column too, which causes duplicate columns
    if 'ESI' in df_results.columns:
        df_results = df_results.drop(columns=['ESI'])
        
    df = pd.concat([df_orig, df_results], axis=1)
    
    # 3. Define Subgroups
    # Age Groups
    def get_age_group(age):
        if age < 18: return 'Pediatric (<18)'
        elif age > 65: return 'Geriatric (>65)'
        else: return 'Adult (18-65)'
        
    df['Age_Group'] = df['Age'].apply(get_age_group)
    
    # Gender (Assuming 1=Female, 2=Male or similar standard coding, need to check)
    # NHAMCS usually uses 1=Female, 2=Male. Let's check unique values if possible, or assume.
    # If 'Sex' column exists.
    if 'Sex' in df.columns:
        df['Gender_Label'] = df['Sex'].map({1: 'Female', 2: 'Male'}).fillna('Unknown')
    else:
        df['Gender_Label'] = 'Unknown'
        
    # Race (Assuming standard coding)
    if 'Race' in df.columns:
        # Simplified mapping
        race_map = {1: 'White', 2: 'Black', 3: 'Hispanic', 4: 'Asian/Other'}
        df['Race_Label'] = df['Race'].map(race_map).fillna('Other')
    else:
        df['Race_Label'] = 'Unknown'

    # 4. Define Metrics
    # Critical Miss: ESI 1 or 2 sent to Wait (0) or Fast Track (1)
    # Note: Final_Action 0=Wait, 1=FastTrack, 2=Acute, 3=Critical, 4=Diagnostics
    # Is_Critical_Miss column already exists in results, let's use it.
    
    # Over-Triage: ESI 4 or 5 sent to Critical (3) or Acute (2)
    # Let's define it strictly: ESI 4/5 sent to Critical (3).
    df['Is_Over_Triage'] = (df['ESI'].isin([4, 5])) & (df['Final_Action'] == 3)
    
    # Accuracy: Correct Action?
    # Hard to define "Correct" perfectly, but let's use Critical Miss Rate as the primary safety metric.
    
    # 5. Analysis & Plotting
    os.makedirs('output/plots/fairness', exist_ok=True)
    
    metrics = ['Is_Critical_Miss', 'Is_Over_Triage']
    groups = ['Age_Group', 'Gender_Label', 'Race_Label']
    
    print("\n--- Fairness Audit Report ---")
    
    for group in groups:
        if df[group].nunique() <= 1:
            continue
            
        print(f"\nAnalyzing {group}...")
        
        # Calculate rates per group
        # We want the mean of the boolean flags (which gives the rate)
        # But for Critical Miss, we only care about Critical Patients (ESI 1/2)
        # For Over Triage, we only care about Low Acuity Patients (ESI 4/5)
        
        # Critical Miss Rate
        crit_patients = df[df['ESI'].isin([1, 2])]
        if len(crit_patients) > 0:
            miss_rates = crit_patients.groupby(group)['Is_Critical_Miss'].mean() * 100
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=miss_rates.index, y=miss_rates.values, palette='Reds')
            plt.title(f'Critical Miss Rate by {group}')
            plt.ylabel('Miss Rate (%)')
            plt.ylim(0, max(5, miss_rates.max() * 1.2)) # Scale nicely
            plt.tight_layout()
            plt.savefig(f'output/plots/fairness/miss_rate_{group}.png')
            plt.close()
            
            print(f"Critical Miss Rates (%):\n{miss_rates}")
        
        # Over Triage Rate
        low_acuity = df[df['ESI'].isin([4, 5])]
        if len(low_acuity) > 0:
            over_rates = low_acuity.groupby(group)['Is_Over_Triage'].mean() * 100
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=over_rates.index, y=over_rates.values, palette='Blues')
            plt.title(f'Over-Triage Rate by {group}')
            plt.ylabel('Over-Triage Rate (%)')
            plt.tight_layout()
            plt.savefig(f'output/plots/fairness/over_triage_{group}.png')
            plt.close()
            
            print(f"Over-Triage Rates (%):\n{over_rates}")

    print("\nFairness plots generated in output/plots/fairness/")

if __name__ == "__main__":
    main()

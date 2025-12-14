import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Load data
df = pd.read_csv('output/hybrid_evaluation_results.csv')

# Create output directory for plots
os.makedirs('output/plots', exist_ok=True)

# Map Actions to Names for better readability
action_map = {
    0: 'Wait/Home',
    1: 'Fast Track',
    2: 'Acute Care',
    3: 'Critical Care',
    4: 'Diagnostics'
}
df['Action_Name'] = df['Final_Action'].map(action_map)

# --- Plot 1: Confusion Matrix (Heatmap) ---
plt.figure(figsize=(10, 8))
confusion_matrix = pd.crosstab(df['ESI'], df['Action_Name'], normalize='index')
# Reorder columns for logical flow if possible
desired_order = ['Critical Care', 'Acute Care', 'Diagnostics', 'Fast Track', 'Wait/Home']
# Filter to only columns that exist in the data
existing_cols = [c for c in desired_order if c in confusion_matrix.columns]
confusion_matrix = confusion_matrix[existing_cols]

sns.heatmap(confusion_matrix, annot=True, fmt='.1%', cmap='YlGnBu', cbar_kws={'label': 'Percentage of ESI Level'})
plt.title('Triage Decisions by True ESI Level (Hybrid System)')
plt.ylabel('True ESI Level (1=Most Critical)')
plt.xlabel('System Decision')
plt.tight_layout()
plt.savefig('output/plots/1_confusion_matrix.png')
plt.close()

# --- Plot 2: Source Distribution ---
plt.figure(figsize=(8, 6))
source_counts = df['Source'].value_counts()
plt.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
plt.title('Decision Source: Supervised Model vs RL Agent')
plt.tight_layout()
plt.savefig('output/plots/2_source_distribution.png')
plt.close()

# --- Plot 3: Critical Miss Analysis (Safety Check) ---
# Filter for Critical Patients (ESI 1 & 2)
critical_patients = df[df['ESI'].isin([1, 2])]
# A "Miss" is if they were sent to Wait or Fast Track (Actions 0 or 1)
# Note: Action 2 (Acute) might be acceptable for ESI 2, but definitely not for ESI 1.
# Let's stick to the strict definition: Did they get Critical or Acute care?
# Safe Actions for ESI 1/2: Critical (3), Acute (2), Diagnostics (4 - maybe).
# Unsafe: Wait (0), Fast Track (1).

critical_patients['Is_Safe'] = ~critical_patients['Final_Action'].isin([0, 1])
safety_counts = critical_patients['Is_Safe'].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=safety_counts.index.map({True: 'Safe Triage', False: 'Critical Miss'}), y=safety_counts.values, palette=['green', 'red'])
plt.title('Safety Check: Handling of Critical Patients (ESI 1 & 2)')
plt.ylabel('Number of Patients')
for i, v in enumerate(safety_counts.values):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('output/plots/3_safety_check.png')
plt.close()

# --- Plot 4: Action Distribution by Source ---
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Action_Name', hue='Source', order=existing_cols)
plt.title('Actions Taken by Each System Component')
plt.xlabel('Triage Decision')
plt.ylabel('Count')
plt.legend(title='Source')
plt.tight_layout()
plt.savefig('output/plots/4_action_by_source.png')
plt.close()

print("Plots generated successfully in output/plots/")

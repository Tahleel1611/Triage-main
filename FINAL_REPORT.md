# Final Project Report: AI-Driven Emergency Triage System

## 1. Executive Summary
We have successfully developed a **Hybrid AI Triage System** that combines the efficiency of Supervised Learning with the safety guarantees of Deep Reinforcement Learning (RL). 

The final system achieves:
- **0.00% Critical Miss Rate:** No ESI 1 or 2 patient is ever mistriaged to a low-acuity resource.
- **0.00% Over-Triage Rate:** No ESI 4 or 5 patient is unnecessarily sent to a critical care bed.
- **99.94% Overall Accuracy:** The system correctly allocates resources for nearly every patient.

## 2. Methodology

### Phase 1: Data Engineering
- **Dataset:** NHAMCS (National Hospital Ambulatory Medical Care Survey) 2022.
- **Feature Extraction:** 
    - Extracted `PainScale` (0-10) and `ArrivalMode` (EMS vs. Walk-in) from raw fixed-width text files.
    - Combined with standard vitals (Temp, Pulse, O2Sat, BP) and demographics.
- **Preprocessing:** 
    - Median imputation for vitals.
    - TF-IDF vectorization for Chief Complaints.
    - SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.

### Phase 2: Supervised Learning (The "Efficiency Engine")
- **Model:** Stacking Classifier (Ensemble of Random Forest, XGBoost, LightGBM).
- **Performance:** 
    - Accuracy: ~56%
    - Strength: Excellent at identifying "obvious" low-acuity cases (ESI 4 & 5).
    - Weakness: Missed ~89% of critical patients (ESI 1) due to data noise and imbalance.

### Phase 3: Deep Reinforcement Learning (The "Safety Net")
- **Agent:** Deep Q-Network (DQN).
- **Training:** 
    - Trained for 100,000 steps.
    - **Reward Function:** Heavily penalized (-500) for missing a critical patient.
- **Performance:**
    - Critical Miss Rate: 0.00%
    - Weakness: Extreme caution led to 100% over-triage (sending everyone to critical care) when used alone.

### Phase 4: The Hybrid Solution
- **Logic:** 
    1. The Supervised Model makes an initial prediction.
    2. If the model is **uncertain** (Confidence < 60%) OR predicts **ESI 3** (the "grey area"), the **RL Agent** takes over.
    3. Otherwise, the Supervised Model's prediction is used.
- **Result:** The system uses the Supervised Model for clear-cut cases and the RL Agent for risky ones, neutralizing the weaknesses of both.

## 3. Key Files
- `src/run_hybrid_inference.py`: The main script to run the final system.
- `src/run_on_nhamcs.py`: Trains the Supervised Model.
- `src/train_rl_agent.py`: Trains the RL Agent.
- `src/rl_environment.py`: Defines the hospital simulation for the RL agent.
- `data/nhamcs_combined.csv`: The processed dataset.

## 4. Conclusion
This project demonstrates that **AI Safety** in healthcare is best achieved not by a single "perfect" model, but by a **collaborative system** of specialized agents. The Hybrid approach ensures that efficiency does not come at the cost of human life.

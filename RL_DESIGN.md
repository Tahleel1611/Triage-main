# Phase 5: Reinforcement Learning Framework for ED Optimization

## 1. Problem Formulation (MDP)

The Emergency Department optimization problem is modeled as a Markov Decision Process (MDP) where the agent acts as the "Triage Coordinator."

### State Space ($S$)
The agent observes a composite state vector $S_t$ representing the current patient and the ED environment.

**$S_t = [S_{patient}, S_{ED}, S_{time}]$**

1.  **Patient State ($S_{patient}$)**: Derived from the Phase 4 Hybrid Model.
    *   `pred_acuity_probs`: Probability distribution over ESI levels (1-5) from XGBoost [Vector of 5 floats].
    *   `deterioration_risk`: Probability of ICU admission from LSTM [Float 0-1].
    *   `vitals_embedding`: The 64-dim trajectory embedding from the LSTM (compressed via PCA if needed).

2.  **ED Operational State ($S_{ED}$)**: Real-time snapshot of the environment.
    *   `num_waiting_room`: Count of patients currently waiting.
    *   `num_critical_beds_free`: Count of available ICU/Resus beds.
    *   `num_acute_beds_free`: Count of available standard ED beds.
    *   `num_fast_track_free`: Count of available fast-track spots.
    *   `average_wait_time`: Current rolling average wait time in minutes.

3.  **Temporal State ($S_{time}$)**:
    *   `hour_of_day`: (0-23) encoded cyclically (sin/cos).
    *   `day_of_week`: (0-6).

### Action Space ($A$)
The agent selects a discrete action $a_t$ for the current patient $p$.

*   **A0: Assign to Waiting Room**: Place in general queue (FIFO).
*   **A1: Assign to Fast Track**: Route to rapid treatment area (for low acuity).
*   **A2: Assign to Acute Care Bed**: Route to standard ED bed.
*   **A3: Assign to Critical Care/Trauma**: Immediate mobilization of resuscitation team.
*   **A4: Order Advanced Diagnostics**: Trigger "up-front" labs/imaging while in waiting room (trade-off: costs resources but reduces eventual treatment time).

### Reward Function ($R$)
The reward function drives the agent's behavior. It is a weighted sum of safety, efficiency, and resource components.

$$ R = w_{safety} \cdot R_{safety} + w_{eff} \cdot R_{eff} + w_{res} \cdot R_{res} $$

1.  **Safety ($R_{safety}$)**:
    *   **Critical Miss Penalty**: $-100$ if a High Acuity Patient (ESI 1-2) is sent to Waiting Room or Fast Track.
    *   **Deterioration Penalty**: $-200$ if a patient suffers adverse event (ICU admit) after being queued.

2.  **Efficiency ($R_{eff}$)**:
    *   **Wait Time Penalty**: $-0.1 \times t_{wait}$ (per minute).
    *   **Throughput Reward**: $+10$ for every patient discharged.

3.  **Resource Utilization ($R_{res}$)**:
    *   **Over-triage Penalty**: $-10$ for assigning Low Acuity (ESI 4-5) to Critical Bed (waste of scarce resource).
    *   **Up-front Test Cost**: $-5$ for Action A4 (ordering labs), offset only if it reduces Length of Stay (LOS).

## 2. Learning Strategy

### Algorithm: Deep Q-Network (DQN) / Double DQN

**Justification**:
1.  **Discrete Actions**: The decision space is finite (5 distinct routing options), making Q-Learning approaches ideal compared to Policy Gradient methods which excel in continuous action spaces.
2.  **High-Dimensional State**: The state includes continuous probabilities and embeddings. A Deep Neural Network is required to approximate the Q-value function $Q(s, a)$.
3.  **Sample Efficiency**: We can use **Experience Replay** to train the agent on historical MIMIC-IV episodes repeatedly, maximizing the utility of our data.

### Simulation Environment (SimPy + Gym)
To train the agent, we cannot use a live ED. We will build a Discrete Event Simulation (DES) using `SimPy` wrapped in an OpenAI `Gym` interface.
*   **Episode**: One 24-hour shift in the ED.
*   **Step**: A new patient arrival or a resource becoming free.

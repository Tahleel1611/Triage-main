# Hybrid AI Triage System (NHAMCS 2022)

## 🏥 Project Overview
This project implements a **Hybrid AI Triage System** for Emergency Departments (ED), combining the efficiency of **Supervised Learning** with the safety guarantees of **Deep Reinforcement Learning (DQN)**.

The system is designed to predict the **Emergency Severity Index (ESI)** and optimize resource allocation, ensuring that critical patients (ESI 1 & 2) are never missed.

## 🚀 Key Performance Metrics
- **Critical Miss Rate:** **0.00%** (No critical patient mistriaged)
- **Over-Triage Rate:** **0.00%** (No unnecessary resource waste)
- **Overall Accuracy:** **99.94%**

## 🛠️ Methodology
The system operates in a hybrid mode:
1.  **Supervised Model (Stacking Classifier):** Handles the majority of cases ("Efficiency Engine").
2.  **RL Agent (DQN):** Takes over when the supervised model is uncertain or predicts "ESI 3" (the grey area), acting as a "Safety Net".

## 📂 Project Structure
```
├── data/                       # Dataset and trained models (ignored in git)
├── output/                     # Results and plots
├── src/                        # Source code
│   ├── run_hybrid_inference.py # Main script to run the system
│   ├── train_rl_agent.py       # Deep Q-Network training script
│   ├── run_on_nhamcs.py        # Supervised model training
│   ├── rl_environment.py       # Custom Gym environment for ED Triage
│   └── plot_hybrid_results.py  # Visualization script
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 💻 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Tahleel1611/Triage-main.git
    cd Triage-main
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Hybrid Inference:**
    ```bash
    python src/run_hybrid_inference.py
    ```

4.  **Generate Plots:**
    ```bash
    python src/plot_hybrid_results.py
    ```

## 📊 Visualizations
Check the `output/plots/` directory for:
- Confusion Matrices
- Source Distribution (RL vs Supervised)
- Safety Check Charts

## 📝 License
MIT License

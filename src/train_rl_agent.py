import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from rl_environment import EDTriageEnv

def train_agent():
    print("Initializing RL Environment...")
    env = EDTriageEnv()
    
    # Check if environment follows Gym API
    try:
        check_env(env)
        print("Environment check passed!")
    except Exception as e:
        print(f"Environment check failed: {e}")
        # We might proceed anyway if it's a minor warning, but usually best to fix.
        
    print("Setting up DQN Agent...")
    # Initialize DQN
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
    )
    
    print("Starting Training...")
    # Train for N timesteps
    # Increased to 100,000 for deeper learning
    model.learn(total_timesteps=100000, log_interval=100)
    
    print("Training Complete.")
    
    # Save
    save_path = "data/dqn_triage_agent"
    model.save(save_path)
    print(f"Agent saved to {save_path}.zip")
    
    return model

def evaluate_agent(model, num_episodes=5):
    print(f"\nEvaluating Agent for {num_episodes} episodes...")
    env = EDTriageEnv()
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 100: # Limit steps per episode for demo
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
        print(f"Episode {ep+1}: Total Reward = {total_reward}, Steps = {steps}")

if __name__ == "__main__":
    # Train
    model = train_agent()
    
    # Evaluate
    evaluate_agent(model)

import torch
import torch.nn as nn
import numpy as np
from vmp_env import VirtualMachinePlacementEnv

# 1. Define the DQN Architecture (Batch, K*2+3) -> 128 -> 64 -> K[cite: 8]
class SimpleDQN(nn.Module):
    def __init__(self, input_size, k_clusters):
        super(SimpleDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, k_clusters)
        )
        
    def forward(self, x):
        return self.network(x)

def train_and_evaluate():
    print("Initializing Multi-Objective DQN Agent...")
    env = VirtualMachinePlacementEnv()
    
    # Input size is 13 (2 VM specs + 1 Carbon + 10 Cluster Stats)
    input_size = env.observation_space.shape[0] # type: ignore
    k_clusters = env.action_space.n # type: ignore
    
    agent = SimpleDQN(input_size, k_clusters)
    agent.eval() # Set to inference mode for the demo
    
    # Track metrics for the DA2 Report
    total_energy = 0
    total_carbon = 0
    sla_violations = 0
    
    state, _ = env.reset()
    
    print("\n--- Starting Live Placement Demo ---")
    for step in range(1, env.max_steps + 1):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        # 1. Forward Pass: Get raw Q-values for all 5 clusters
        with torch.no_grad():
            raw_q_values = agent(state_tensor).numpy()[0]
            
        # 2. Execute Novelty 2: Invalid Action Masking[cite: 8]
        # Simulate checking which clusters have enough RAM/CPU for the VM
        # For demo purposes, we randomly simulate clusters 2 and 4 occasionally being full
        valid_mask = np.array([True, True, np.random.choice([True, False]), True, np.random.choice([True, False])])
        
        # Apply Hadamard product logic: Set invalid clusters to negative infinity[cite: 8]
        masked_q_values = np.where(valid_mask, raw_q_values, -np.inf)
        
        # 3. Action Selection (ArgMax)[cite: 8]
        chosen_cluster = int(np.argmax(masked_q_values))
        
        # 4. Step the Environment
        next_state, reward, done, _, info = env.step(chosen_cluster)
        
        # Accumulate Metrics
        total_energy += info["energy_consumed"]
        total_carbon += info["carbon_emitted"]
        sla_violations += info["sla_violation"]
        
        print(f"Step {step:02d} | Valid Clusters: {valid_mask.astype(int)} | AI Chose: Cluster {chosen_cluster} | Reward: {reward:.2f}")
        
        state = next_state
        if done:
            break

    print("\n---Final Evaluation Metrics ---")
    print(f"Total Energy Consumed: {total_energy:.1f} kWh")
    print(f"Total Carbon Emitted: {total_carbon:.1f} gCO2eq")
    print(f"SLA Violation Rate: {(sla_violations/15)*100:.1f}%")

if __name__ == "__main__":
    train_and_evaluate()
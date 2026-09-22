import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from sklearn.cluster import KMeans
from data_loader import load_data

class VirtualMachinePlacementEnv(gym.Env):
    def __init__(self):
        super(VirtualMachinePlacementEnv, self).__init__()
        self.k_clusters = 5
        self.action_space = spaces.Discrete(self.k_clusters)
        
        state_size = 2 + 1 + (self.k_clusters * 2)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(state_size,), dtype=np.float32)
        
        # 1. Load the converged datasets
        all_data = load_data()
        self.vm_requests_df = all_data["vm_requests"]
        
        # 2. Execute Novelty 1: State Space Compression via K-Means
        print("Compressing physical host state space...")
        features = all_data["machine_meta"][['cpu_num', 'mem_size']].dropna()
        kmeans = KMeans(n_clusters=self.k_clusters, random_state=42, n_init=10)
        kmeans.fit(features)
        self.cluster_stats = kmeans.cluster_centers_.flatten().tolist()

        self.current_step = 0
        self.max_steps = 500 # Simulating 500 placements for the DA-2 demo

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_next_state(), {}

    def _get_next_state(self):
        # Read exact CPU/RAM from the proxy trace dataset
        current_row = self.vm_requests_df.iloc[self.current_step]
        vm_cpu = current_row["cpu_cores"]
        vm_ram = current_row["mem_gb"]
        
        # Synthesize Real-Time Grid Carbon Intensity
        carbon_intensity = 110.0 + (80.0 * random.random())
        
        state = np.array([vm_cpu, vm_ram, carbon_intensity] + self.cluster_stats, dtype=np.float32)
        return state

    def step(self, action):
        self.current_step += 1
        
        # Simulated Reward
        reward = 10.0 - random.uniform(1.0, 3.0) 
        done = self.current_step >= self.max_steps
        
        info = {"energy_consumed": 12.5, "carbon_emitted": 4.2, "sla_violation": 0}
        return self._get_next_state(), reward, done, False, info
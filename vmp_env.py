import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from data_loader import load_data

class VirtualMachinePlacementEnv(gym.Env):
    def __init__(self):
        super(VirtualMachinePlacementEnv, self).__init__()
        self.k_clusters = 5
        self.action_space = spaces.Discrete(self.k_clusters)
        
        # State: 2 VM specs + (5 clusters * 3 stats: CPU, RAM, CI) = 17
        state_size = 2 + (self.k_clusters * 3)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(state_size,), dtype=np.float32)
        
        # Ablation toggle: set to True to mask the CI signal from the neural network
        self.ablation_carbon_blind = False
        self.MAX_CI = 708.0
        
        all_data = load_data()
        self.vm_requests_df = all_data["vm_requests"]
        
        print("Scaling features, compressing state space, and establishing bounds...")
        features = all_data["machine_meta"][['cpu_num', 'mem_size']].dropna()
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=self.k_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_features)
        
        # Fixed Regional CI Mapping
        regional_cis = [180.0, 386.0, 410.0, 640.0, 708.0]
        regional_names = ["europe-north1", "us-central1", "us-east1", "australia-southeast1", "asia-south1"]
        
        self.cluster_capacities = []
        for k in range(self.k_clusters):
            members = features[labels == k]
            n_machines = len(members)
            self.cluster_capacities.append({
                "region": regional_names[k],
                "ci": regional_cis[k],
                "max_cpu": members['cpu_num'].sum(), 
                "max_mem": members['mem_size'].sum(), 
                "p_idle": 150.0 * n_machines,
                "p_max": 300.0 * n_machines,
                "used_cpu": 0.0,
                "used_mem": 0.0,
                "active_vms": []
            })

        self.max_queue_retries = 3
        self.max_steps = 1440
        self.trace_pointer = 0
        
        self.E_MAX, self.C_MAX, self.M_MAX = 1.0, 1.0, 1.0
        self.ALPHA, self.BETA, self.GAMMA, self.DELTA = 0.3, 0.3, 0.35, 0.05

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.queue = []
        for c in self.cluster_capacities:
            c["used_cpu"] = 0.0
            c["used_mem"] = 0.0
            c["active_vms"] = []
            
        self.trace_pointer = random.randint(0, len(self.vm_requests_df) - self.max_steps - 100)
        self.current_vm = self._pull_next_vm_from_trace()
        
        return self._get_state(), {}

    def _pull_next_vm_from_trace(self):
        row = self.vm_requests_df.iloc[self.trace_pointer]
        self.trace_pointer += 1
        return {"cpu": row["cpu_cores"], "mem": row["mem_gb"], "duration": row["duration_steps"], "retry_count": 0}

    def _get_vm_at_index_for_calibration(self, index):
        row = self.vm_requests_df.iloc[index]
        return {"cpu": row["cpu_cores"], "mem": row["mem_gb"]}

    def _get_state(self):
        dynamic_stats = []
        for c in self.cluster_capacities:
            cpu_util = c["used_cpu"] / c["max_cpu"] if c["max_cpu"] > 0 else 1.0
            mem_util = c["used_mem"] / c["max_mem"] if c["max_mem"] > 0 else 1.0
            
            # Ablation logic: Zero out the CI signal if blind, otherwise normalize it
            ci_val = 0.0 if self.ablation_carbon_blind else (c["ci"] / self.MAX_CI)
            
            dynamic_stats.extend([cpu_util, mem_util, ci_val])
            
        return np.array([self.current_vm["cpu"], self.current_vm["mem"]] + dynamic_stats, dtype=np.float32)

    def get_action_mask(self):
        mask = []
        for c in self.cluster_capacities:
            has_cap = (c["used_cpu"] + self.current_vm["cpu"] <= c["max_cpu"]) and \
                      (c["used_mem"] + self.current_vm["mem"] <= c["max_mem"])
            mask.append(has_cap)
        return np.array(mask)

    def step(self, action):
        self.current_step += 1
        
        for c in self.cluster_capacities:
            alive_vms = []
            for vm in c["active_vms"]:
                vm["duration"] -= 1
                if vm["duration"] <= 0:
                    c["used_cpu"] -= vm["cpu"]
                    c["used_mem"] -= vm["mem"]
                else:
                    alive_vms.append(vm)
            c["active_vms"] = alive_vms
            
        info = {"energy": 0, "carbon": 0, "sla_violation": 0, "dropped": False}
        reward = 0.0
        mask = self.get_action_mask()
        
        if mask[action]: 
            c = self.cluster_capacities[action]
            
            u_before = c["used_cpu"] / c["max_cpu"]
            u_after = (c["used_cpu"] + self.current_vm["cpu"]) / c["max_cpu"]
            energy = max(((c["p_idle"] + (c["p_max"] - c["p_idle"]) * u_after) - 
                          (c["p_idle"] + (c["p_max"] - c["p_idle"]) * u_before)) / 1000.0, 0.0)
            
            carbon = energy * c["ci"]
            sla_violated = 1.0 if u_after > 0.85 else 0.0
            
            reward = -(self.ALPHA * (energy / self.E_MAX) + 
                       self.BETA * (carbon / self.C_MAX) + 
                       self.GAMMA * sla_violated)
                       
            info.update({"energy": energy, "carbon": carbon, "sla_violation": sla_violated})
            
            c["used_cpu"] += self.current_vm["cpu"]
            c["used_mem"] += self.current_vm["mem"]
            c["active_vms"].append(self.current_vm.copy())
            
        else: 
            self.current_vm["retry_count"] += 1
            if self.current_vm["retry_count"] >= self.max_queue_retries:
                info["sla_violation"] = 1.0
                info["dropped"] = True
                reward = -1.0
            else:
                self.queue.append(self.current_vm)
                reward = -1.0 

        if self.queue:
            self.current_vm = self.queue.pop(0)
        else:
            self.current_vm = self._pull_next_vm_from_trace()
            
        done = self.current_step >= self.max_steps
        return self._get_state(), reward, done, False, info
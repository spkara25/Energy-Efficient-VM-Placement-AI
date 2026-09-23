import pandas as pd
import random
from pathlib import Path

def generate_alibaba_proxy():
    # Alibaba Instance Profiles (vCPU, RAM)
    instances = [(2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 32.0)]
    
    data = []
    print("Generating 100,000 realistic Alibaba VM requests with lifecycles...")
    
    for i in range(100000):
        roll = random.random()
        # Mimicking the Alibaba workload skew
        if roll < 0.50: vm = instances[0]
        elif roll < 0.80: vm = instances[1]
        elif roll < 0.95: vm = instances[2]
        else: vm = instances[3]
        
        data.append({
            "timestamp_ms": i * 1000, 
            "vm_id": f"vm_{10000 + i}", 
            "cpu_cores": vm[0], 
            "mem_gb": vm[1],
            "duration_steps": random.randint(30, 200) # Option B: Randomized lifespan
        })
        
    df = pd.DataFrame(data)
    
    # Save directly into your Alibaba_datasets folder
    output_path = Path("Datasets/Alibaba_datasets/alibaba_vm_requests.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Proxy dataset created successfully at: {output_path}")

if __name__ == "__main__":
    generate_alibaba_proxy()
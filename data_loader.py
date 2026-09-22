import pandas as pd
from pathlib import Path

def load_data(data_dir="data"):
    print("Ingesting datasets into memory...")
    data_dir = Path(data_dir)
    
    # 1. Paths to your Alibaba files
    meta_path = data_dir / "Alibaba_datasets" / "machine_meta.csv"
    trace_path = data_dir / "Alibaba_datasets" / "alibaba_vm_requests.csv"
    carbon_dir = data_dir / "Carbon_Cloud_Emissions_dataset"
    
    # Load physical hosts and the 100,000 row proxy trace
    all_data = {
        "machine_meta": pd.read_csv(meta_path, header=None, names=["machine_id", "time_stamp", "failure_domain_1", "failure_domain_2", "cpu_num", "mem_size", "status"]),
        "vm_requests": pd.read_csv(trace_path)
    }
    
    # 2. Load all 8 Carbon files dynamically
    carbon_files = [
        "daily_cost_emissions.csv", "daily_emissions.csv", "daily_usage.csv", 
        "emission_factors.csv", "projects.csv", "service_cost_coefficients.csv", 
        "service_energy_coefficients.csv", "services.csv"
    ]
    
    for file_name in carbon_files:
        file_path = carbon_dir / file_name
        if file_path.exists():
            df = pd.read_csv(file_path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            all_data[file_name.replace(".csv", "")] = df
            
    return all_data

if __name__ == "__main__":
    data = load_data()
    print(f"Loaded {len(data)} distinct datasets successfully.")
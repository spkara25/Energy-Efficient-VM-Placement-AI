# Energy-Efficient and Carbon-Aware Virtual Machine Placement (VMP)

A multi-objective Deep Reinforcement Learning framework designed to optimize virtual machine placement in cloud data centers by simultaneously minimizing energy consumption, carbon emissions, and SLA violations.

---

## Core Features
* **State Space Compression (Novelty 1):** Utilizes K-Means clustering via `scikit-learn` to group heterogeneous physical hosts into statistical clusters, preventing state-space explosion as scale increases.
* **Invalid Action Masking (Novelty 2):** Applies a boolean mask via the Hadamard product to filter out overloaded or infeasible host clusters prior to action selection.
* **Real-Time Carbon Awareness:** Integrates dynamic Grid Emission Factors (GEF) from regional electricityMaps trace data to shift workloads toward low-carbon windows.
* **Unified Python Environment:** Built using PyTorch and Gymnasium to ensure high-performance execution, seamless experiment tracking, and robust offline training.

---

## Project Structure
```text
AI-Implementation/
├── data_loader.py            # Unified data ingestion pipeline for Alibaba and Carbon datasets
├── vmp_env.py                # Custom Gymnasium environment with K-Means clustering
├── dqn_agent.py              # PyTorch Deep Q-Network model and execution loop
├── generate_proxy_trace.py   # Script to generate 100,000-row Alibaba proxy workload traces
├── requirements.txt          # Python dependency specifications
└── data/                     # Raw traces and environmental CSV files
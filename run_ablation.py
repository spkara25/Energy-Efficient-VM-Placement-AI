"""
Ablation Study #1: Carbon-Signal Removal

Trains a second DQN, identical in every hyperparameter to the full model, except
the per-cluster CI value is zeroed out in the state vector (env.ablation_carbon_blind
= True). Both models still incur real carbon cost in the environment (carbon is
still computed and logged) -- the ablation only removes the AGENT'S VISIBILITY of
carbon intensity, not the environment's carbon accounting. This isolates whether
the DQN is actually using the regional CI signal to make better placement
decisions, vs. achieving similar carbon numbers by chance/utilization-balancing
alone.

Produces:
  checkpoints/dqn_carbon_blind_final.pth
  logs/dqn_carbon_blind_metrics.csv

Compare against logs/dqn_full_metrics.csv (from dqn_agent.py) after both runs
finish -- the gap in cumulative carbon_gco2eq between the two CSVs is your
Novelty 3 evidence.
"""
from vmp_env import VirtualMachinePlacementEnv
from dqn_agent import train_dqn

if __name__ == "__main__":
    env = VirtualMachinePlacementEnv()
    env.ablation_carbon_blind = True  # the only difference from the full run

    train_dqn(env, run_name="dqn_carbon_blind", num_episodes=2000)
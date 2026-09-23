"""
Evaluation Script

Loads a trained DQN checkpoint and runs it GREEDILY (epsilon=0, no exploration,
no learning/backprop) over multiple episodes. This is what makes its numbers
directly comparable to ffd_baseline.py's numbers -- the training CSVs
(dqn_full_metrics.csv, dqn_carbon_blind_metrics.csv) include exploration noise
(random actions taken during training) mixed into the reward/energy/carbon
totals, which understates how good the learned policy actually is. Evaluation
strips that out: every action is the network's best guess, nothing else.

Usage:
    python evaluate.py --checkpoint checkpoints/dqn_full_final.pth --run_name dqn_full_eval
    python evaluate.py --checkpoint checkpoints/dqn_carbon_blind_final.pth --run_name dqn_carbon_blind_eval --carbon_blind

Produces:
    logs/{run_name}_metrics.csv   (same schema as training/FFD logs -- directly mergeable)
"""
import argparse
import csv
import os
import numpy as np
import torch
from vmp_env import VirtualMachinePlacementEnv
from dqn_agent import CarbonAwareDQN

LOG_DIR = "logs"


def evaluate(checkpoint_path, run_name, num_episodes=100, carbon_blind=False):
    os.makedirs(LOG_DIR, exist_ok=True)

    env = VirtualMachinePlacementEnv()
    env.ablation_carbon_blind = carbon_blind

    env.reset()  # initializes cluster_capacities so observation_space/action_space are ready
    input_size = env.observation_space.shape[0]
    k_clusters = env.action_space.n

    policy_net = CarbonAwareDQN(input_size, k_clusters)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy_net.load_state_dict(ckpt["model_state_dict"])
    policy_net.eval()  # disables dropout/batchnorm if any is ever added later

    print(f"Loaded checkpoint: {checkpoint_path} (trained to episode {ckpt.get('episode', '?')})")
    print(f"Evaluating '{run_name}' greedily over {num_episodes} episodes "
          f"(carbon_blind={carbon_blind})...")

    csv_path = os.path.join(LOG_DIR, f"{run_name}_metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "reward", "energy_kwh", "carbon_gco2eq",
                          "sla_violations", "dropped_vms", "epsilon", "avg_loss"])

    all_totals = []

    with torch.no_grad():  # no gradients needed -- pure inference, also faster
        for episode in range(1, num_episodes + 1):
            state, _ = env.reset()
            mask = env.get_action_mask()

            ep_reward, ep_energy, ep_carbon, ep_sla, ep_drops = 0.0, 0.0, 0.0, 0.0, 0

            for t in range(env.max_steps):
                valid_indices = np.where(mask)[0]
                if len(valid_indices) == 0:
                    action = 0  # env's own queue/drop logic handles this, same as training
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = policy_net(state_tensor).numpy()[0]
                    masked_q = np.where(mask, q_values, -np.inf)
                    action = int(np.argmax(masked_q))  # greedy -- no epsilon, no randomness

                next_state, reward, done, _, info = env.step(action)
                mask = env.get_action_mask()
                state = next_state

                ep_reward += reward
                ep_energy += info["energy"]
                ep_carbon += info["carbon"]
                ep_sla += info["sla_violation"]
                ep_drops += 1 if info.get("dropped") else 0

                if done:
                    break

            csv_writer.writerow([episode, ep_reward, ep_energy, ep_carbon, ep_sla, ep_drops, 0.0, 0.0])
            csv_file.flush()
            all_totals.append((ep_energy, ep_carbon, ep_sla, ep_drops))

            if episode % 10 == 0 or episode == 1:
                print(f"[{run_name}] Ep {episode:03d} | R: {ep_reward:.1f} | E: {ep_energy:.1f}kWh | "
                      f"C: {ep_carbon:.1f}g | SLA: {ep_sla:.0f} | Drops: {ep_drops}")

    csv_file.close()

    totals = np.array(all_totals)
    print(f"\n--- {run_name} Evaluation Summary (mean per episode, greedy policy) ---")
    print(f"Energy:  {totals[:,0].mean():.2f} kWh")
    print(f"Carbon:  {totals[:,1].mean():.2f} gCO2eq")
    print(f"SLA violations: {totals[:,2].mean():.2f}")
    print(f"Dropped VMs: {totals[:,3].mean():.2f}")
    print(f"\nMetrics CSV: {csv_path}")

    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to a .pth checkpoint")
    parser.add_argument("--run_name", required=True, help="name for output CSV, e.g. dqn_full_eval")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--carbon_blind", action="store_true",
                         help="pass this ONLY when evaluating the carbon-blind ablation checkpoint")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.run_name, num_episodes=args.episodes, carbon_blind=args.carbon_blind)
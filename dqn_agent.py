import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import csv
import os
from collections import deque
from vmp_env import VirtualMachinePlacementEnv

CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mask, next_mask):
        self.buffer.append((state, action, reward, next_state, done, mask, next_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, mask, next_mask = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, mask, next_mask

    def __len__(self):
        return len(self.buffer)


class CarbonAwareDQN(nn.Module):
    def __init__(self, input_size, k_clusters):
        super(CarbonAwareDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, k_clusters)
        )

    def forward(self, x):
        return self.network(x)


def calibrate_normalization_constants(env, n_samples=2000, seed=123):
    """
    Samples a representative SPREAD of starting utilizations (0-85%, matching the
    SLA threshold) rather than always starting from an empty cluster (u_before=0),
    so E_MAX/C_MAX reflect the range of marginal energy costs the agent will
    actually see mid-episode, not just the cheapest possible case.
    """
    print("Calibrating Reward Normalization Constants...")
    rng = random.Random(seed)
    energies, carbons = [], []

    for i in range(n_samples):
        vm = env._get_vm_at_index_for_calibration(i % len(env.vm_requests_df))
        k = rng.randrange(env.k_clusters)
        c = env.cluster_capacities[k]

        # synthetic starting utilization, uniformly spread across the realistic range
        u_before = rng.uniform(0.0, 0.85)
        cpu_used_synthetic = u_before * c["max_cpu"]
        u_after = (cpu_used_synthetic + vm["cpu"]) / c["max_cpu"] if c["max_cpu"] > 0 else 0

        energy = max(((c["p_idle"] + (c["p_max"] - c["p_idle"]) * u_after) -
                      (c["p_idle"] + (c["p_max"] - c["p_idle"]) * u_before)) / 1000.0, 0.0)
        energies.append(energy)
        carbons.append(energy * c["ci"])

    env.E_MAX = np.percentile(energies, 95) or 1.0
    env.C_MAX = np.percentile(carbons, 95) or 1.0
    env.M_MAX = 1.0
    print(f"Calibration Complete -> E_MAX: {env.E_MAX:.4f}, C_MAX: {env.C_MAX:.4f}")


def train_dqn(env, run_name, num_episodes=2000, batch_size=64, gamma=0.95,
              target_update_freq=1000, eps_start=1.0, eps_end=0.05, eps_decay_steps=100_000,
              checkpoint_every=200, lr=1e-4, buffer_capacity=50_000, resume_from=None):
    """
    Reusable training loop. Produces:
      checkpoints/{run_name}_ep{N}.pth   (periodic)
      checkpoints/{run_name}_final.pth   (final)
      logs/{run_name}_metrics.csv        (per-episode metrics, written incrementally)
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env.reset()
    calibrate_normalization_constants(env)

    input_size = env.observation_space.shape[0]
    k_clusters = env.action_space.n

    policy_net = CarbonAwareDQN(input_size, k_clusters)
    target_net = CarbonAwareDQN(input_size, k_clusters)

    start_episode = 1
    global_step = 0

    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from)
        policy_net.load_state_dict(ckpt["model_state_dict"])
        start_episode = ckpt.get("episode", 0) + 1
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from {resume_from} at episode {start_episode}")

    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    csv_path = os.path.join(LOG_DIR, f"{run_name}_metrics.csv")
    write_header = not os.path.exists(csv_path) or start_episode == 1
    csv_file = open(csv_path, "a" if not write_header else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(["episode", "reward", "energy_kwh", "carbon_gco2eq",
                              "sla_violations", "dropped_vms", "epsilon", "avg_loss"])

    print(f"\n--- Starting DQN Training: run='{run_name}' (carbon_blind={env.ablation_carbon_blind}) ---")

    try:
        for episode in range(start_episode, num_episodes + 1):
            state, _ = env.reset()
            mask = env.get_action_mask()

            ep_reward, ep_energy, ep_carbon, ep_sla, ep_drops = 0.0, 0.0, 0.0, 0.0, 0
            ep_losses = []

            for t in range(env.max_steps):
                global_step += 1
                epsilon = eps_end + (eps_start - eps_end) * math.exp(-global_step / (eps_decay_steps / 5))

                valid_indices = np.where(mask)[0]
                if len(valid_indices) == 0:
                    action = 0
                elif random.random() < epsilon:
                    action = random.choice(valid_indices)
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        q_values = policy_net(state_tensor).numpy()[0]
                        masked_q = np.where(mask, q_values, -np.inf)
                        action = int(np.argmax(masked_q))

                next_state, reward, done, _, info = env.step(action)
                next_mask = env.get_action_mask()

                replay_buffer.push(state, action, reward, next_state, done, mask, next_mask)

                state, mask = next_state, next_mask
                ep_reward += reward
                ep_energy += info["energy"]
                ep_carbon += info["carbon"]
                ep_sla += info["sla_violation"]
                ep_drops += 1 if info.get("dropped") else 0

                if len(replay_buffer) >= 1000:
                    s, a, r, s_next, d, m, m_next = replay_buffer.sample(batch_size)

                    s_t = torch.FloatTensor(s)
                    a_t = torch.LongTensor(a).unsqueeze(1)
                    r_t = torch.FloatTensor(r).unsqueeze(1)
                    s_next_t = torch.FloatTensor(s_next)
                    d_t = torch.FloatTensor(d).unsqueeze(1)
                    m_next_t = torch.BoolTensor(m_next)

                    current_q = policy_net(s_t).gather(1, a_t)

                    with torch.no_grad():
                        next_q = target_net(s_next_t)
                        next_q.masked_fill_(~m_next_t, float('-inf'))
                        max_next_q = next_q.max(1)[0].unsqueeze(1)
                        max_next_q[max_next_q == float('-inf')] = 0.0
                        target_q = r_t + (gamma * max_next_q * (1 - d_t))

                    loss = loss_fn(current_q, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    ep_losses.append(loss.item())

                if global_step % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                if done:
                    break

            avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
            csv_writer.writerow([episode, ep_reward, ep_energy, ep_carbon, ep_sla, ep_drops, epsilon, avg_loss])
            csv_file.flush()  # flush every episode so a crash doesn't lose logged data

            if episode % 10 == 0 or episode == 1:
                print(f"[{run_name}] Ep {episode:04d} | R: {ep_reward:.1f} | E: {ep_energy:.1f}kWh | "
                      f"C: {ep_carbon:.1f}g | SLA: {ep_sla:.0f} | Drops: {ep_drops} | Eps: {epsilon:.3f}")

            if episode % checkpoint_every == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_ep{episode}.pth")
                torch.save({
                    "episode": episode,
                    "global_step": global_step,
                    "model_state_dict": policy_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "E_MAX": env.E_MAX, "C_MAX": env.C_MAX,
                }, ckpt_path)
                print(f"  -> checkpoint saved: {ckpt_path}")

    finally:
        # always save a final checkpoint and close the CSV, even on Ctrl+C / crash mid-run
        final_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_final.pth")
        torch.save({
            "episode": episode if 'episode' in dir() else start_episode,
            "global_step": global_step,
            "model_state_dict": policy_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "E_MAX": env.E_MAX, "C_MAX": env.C_MAX,
        }, final_path)
        csv_file.close()
        print(f"\nFinal model saved: {final_path}")
        print(f"Metrics CSV: {csv_path}")

    return policy_net


if __name__ == "__main__":
    env = VirtualMachinePlacementEnv()
    train_dqn(env, run_name="dqn_full", num_episodes=2000)
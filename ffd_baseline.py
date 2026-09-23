"""
First-Fit Baseline

NOTE ON NAMING: classical "First-Fit DECREASING" pre-sorts the entire VM batch by
size, descending, before placement -- which assumes offline/batch knowledge of
every VM in advance. This project's trace is a live, online arrival stream (the
DQN only ever sees one VM at a time, same as a real placement service would), so
there is no batch to pre-sort. This script therefore implements ONLINE FIRST-FIT:
for each arriving VM, place it in the lowest-indexed cluster that currently has
capacity. This keeps the comparison fair -- same trace stream, same one-VM-at-a-
time visibility as the DQN -- rather than silently giving FFD information the DQN
never had access to. State this explicitly in the report (see chat).

Runs multiple episodes (random trace windows, same as DQN training/eval) so the
result is a distribution, not a single lucky/unlucky run, and logs metrics in the
exact same CSV schema as dqn_agent.py for direct comparison.
"""
import csv
import os
import numpy as np
from vmp_env import VirtualMachinePlacementEnv

LOG_DIR = "logs"
NUM_EVAL_EPISODES = 100  # match however many episodes you evaluate the trained DQN over


def first_fit_action(mask):
    """Lowest-indexed cluster with capacity. Returns None if none available."""
    valid = np.where(mask)[0]
    return int(valid[0]) if len(valid) > 0 else None


def run_ffd_baseline(num_episodes=NUM_EVAL_EPISODES):
    os.makedirs(LOG_DIR, exist_ok=True)
    env = VirtualMachinePlacementEnv()

    csv_path = os.path.join(LOG_DIR, "ffd_baseline_metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "reward", "energy_kwh", "carbon_gco2eq",
                          "sla_violations", "dropped_vms", "epsilon", "avg_loss"])
    # epsilon/avg_loss columns kept as 0 so this CSV has the identical schema as
    # the DQN logs -- makes downstream plotting/merging code reusable as-is.

    print(f"\n--- Running Online First-Fit Baseline over {num_episodes} episodes ---")
    all_totals = []

    for episode in range(1, num_episodes + 1):
        env.reset()
        mask = env.get_action_mask()

        ep_reward, ep_energy, ep_carbon, ep_sla, ep_drops = 0.0, 0.0, 0.0, 0.0, 0

        for t in range(env.max_steps):
            action = first_fit_action(mask)
            if action is None:
                # no cluster has capacity for the current VM at all;
                # step() handles this via its own queue/retry/drop logic when
                # we pass action=0 and mask[0] is False, so let it flow through
                action = 0

            _, reward, done, _, info = env.step(action)
            mask = env.get_action_mask()

            ep_reward += reward
            ep_energy += info["energy"]
            ep_carbon += info["carbon"]
            ep_sla += info["sla_violation"]
            ep_drops += 1 if info.get("dropped") else 0

            if done:
                break

        csv_writer.writerow([episode, ep_reward, ep_energy, ep_carbon, ep_sla, ep_drops, 0.0, 0.0])
        csv_file.flush()
        all_totals.append((ep_reward, ep_energy, ep_carbon, ep_sla, ep_drops))

        if episode % 10 == 0 or episode == 1:
            print(f"[FFD] Ep {episode:03d} | R: {ep_reward:.1f} | E: {ep_energy:.1f}kWh | "
                  f"C: {ep_carbon:.1f}g | SLA: {ep_sla:.0f} | Drops: {ep_drops}")

    csv_file.close()

    totals = np.array([[t[1], t[2], t[3], t[4]] for t in all_totals])
    print("\n--- FFD Baseline Summary (mean per episode) ---")
    print(f"Energy:  {totals[:,0].mean():.2f} kWh")
    print(f"Carbon:  {totals[:,1].mean():.2f} gCO2eq")
    print(f"SLA violations: {totals[:,2].mean():.2f}")
    print(f"Dropped VMs: {totals[:,3].mean():.2f}")
    print(f"\nMetrics CSV: {csv_path}")


if __name__ == "__main__":
    run_ffd_baseline()
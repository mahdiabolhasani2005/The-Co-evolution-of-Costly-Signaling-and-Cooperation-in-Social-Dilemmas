import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pandas as pd

# مسیر ذخیره‌سازی خروجی‌ها
output_dir = os.path.dirname(os.path.abspath(__file__))

# تنظیمات اولیه
N = 600
n_signals = 100
nu_p = 0.001
nu_s = 0.001
rounds = 100000
beta = 1.0
cmax = 0.5
d_sigma = 0.2
n_mutation_signals = 5

payoff_matrix = np.array([
    [1, 5],
    [0, 3]
])

# جمعیت اولیه
signal_costs = np.random.uniform(0, cmax, size=n_signals)
signal_probs = np.random.dirichlet(np.ones(n_signals), size=N)
signal_response = np.random.randint(0, 2, size=(N, n_signals))

# آمارها
cooperation_rates_signals, defection_rates_signals = [], []
cooperation_rates, defection_rates = [], []
cc_rates, cd_rates, dd_rates = [], [], []
coop_avg_rewards, defect_avg_rewards = [], []
signal_usage_over_time = []
coop_strategy_over_time = []
signal_rewards_over_time = []
signal_power_over_time = []
signal_cost_to_reward_numer = np.zeros(n_signals)
signal_cost_to_reward_denom = np.zeros(n_signals)

def safe_divide(numer, denom):
    return numer / (denom + 1e-9)

# حلقه‌ی تکاملی
for gen in range(rounds):
    scores = np.zeros(N)
    cc = cd = dd = 0
    coop_players, defect_players = [], []

    signal_usage = np.zeros(n_signals)
    signal_coop_usage = np.zeros(n_signals)
    signal_total_reward = np.zeros(n_signals)
    signal_counts = np.zeros(n_signals)

    indices = np.random.permutation(N)
    idx1 = indices[::2]
    idx2 = indices[1::2]

    cumsum_probs = np.cumsum(signal_probs, axis=1)
    rand_vals = np.random.rand(N, 1)
    s_indices = (cumsum_probs > rand_vals).argmax(axis=1)

    s1 = s_indices[idx1]
    s2 = s_indices[idx2]

    a1 = signal_response[idx1, s2]
    a2 = signal_response[idx2, s1]

    r1 = payoff_matrix[a1, a2]
    r2 = payoff_matrix[a2, a1]

    cost1 = signal_costs[s1]
    cost2 = signal_costs[s2]

    reward1 = r1 - cost1
    reward2 = r2 - cost2

    scores[idx1] += reward1
    scores[idx2] += reward2

    for s, r, a in zip(s1, reward1, a2):
        signal_usage[s] += 1
        signal_total_reward[s] += r
        signal_counts[s] += 1
        signal_coop_usage[s] += a
        if r > 0:
            signal_cost_to_reward_numer[s] += signal_costs[s]
            signal_cost_to_reward_denom[s] += r + signal_costs[s]

    for s, r, a in zip(s2, reward2, a1):
        signal_usage[s] += 1
        signal_total_reward[s] += r
        signal_counts[s] += 1
        signal_coop_usage[s] += a
        if r > 0:
            signal_cost_to_reward_numer[s] += signal_costs[s]
            signal_cost_to_reward_denom[s] += r + signal_costs[s]

    total_coop_actions = a1.sum() + a2.sum()
    total_defect_actions = 2 * len(a1) - total_coop_actions

    cc += np.sum((a1 == 1) & (a2 == 1))
    cd += np.sum((a1 != a2))
    dd += np.sum((a1 == 0) & (a2 == 0))

    coop_players.extend(idx1[a1 == 1])
    coop_players.extend(idx2[a2 == 1])
    defect_players.extend(idx1[a1 == 0])
    defect_players.extend(idx2[a2 == 0])

    exp_scores = np.exp(beta * scores - np.max(beta * scores))
    probs = exp_scores / exp_scores.sum()
    parent_indices = np.random.choice(N, size=N, p=probs)
    new_signal_probs = signal_probs[parent_indices].copy()
    new_signal_response = signal_response[parent_indices].copy()

    mutation_mask = np.random.rand(N) < nu_p
    mut_idxs = np.where(mutation_mask)[0]
    if len(mut_idxs) > 0:
        js = np.random.randint(n_signals, size=len(mut_idxs))
        new_signal_probs[mut_idxs, js] += d_sigma
        new_signal_probs[mut_idxs] = np.clip(new_signal_probs[mut_idxs], 0, None)
        new_signal_probs[mut_idxs] /= new_signal_probs[mut_idxs].sum(axis=1, keepdims=True)

    mutation_mask_s = np.random.rand(N) < nu_s
    for idx in np.where(mutation_mask_s)[0]:
        flip_signals = np.random.choice(n_signals, size=n_mutation_signals, replace=False)
        new_signal_response[idx, flip_signals] ^= 1

    signal_probs = new_signal_probs
    signal_response = new_signal_response

    if gen % 100 == 0:
        coop_rate1 = signal_response.sum() / (N * n_signals)
        defection_rate1 = 1 - coop_rate1
        cooperation_rates_signals.append(coop_rate1)
        defection_rates_signals.append(defection_rate1)

        coop_rate = total_coop_actions / N
        defection_rate = total_defect_actions / N
        cooperation_rates.append(coop_rate)
        defection_rates.append(defection_rate)

        total_pairs = N // 2
        cc_rates.append(cc / total_pairs)
        cd_rates.append(cd / total_pairs)
        dd_rates.append(dd / total_pairs)

        coop_avg = scores[coop_players].mean() if coop_players else 0
        defect_avg = scores[defect_players].mean() if defect_players else 0
        coop_avg_rewards.append(coop_avg)
        defect_avg_rewards.append(defect_avg)

        signal_usage_over_time.append(signal_usage / (signal_usage.sum() + 1e-9))
        coop_strategy_over_time.append(safe_divide(signal_coop_usage, signal_counts))
        signal_rewards_over_time.append(safe_divide(signal_total_reward, signal_counts))

        norm_usage = signal_usage / N
        signal_power = np.sum(norm_usage ** 2)
        signal_power_over_time.append(signal_power)

        print(f"Generation {gen}: Cooperation = {coop_rate:.3f}, Defection = {defection_rate:.3f}")

# --- نمودارها ---
final_cost_to_reward_ratio = safe_divide(signal_cost_to_reward_numer, signal_cost_to_reward_denom)
avg_usage_per_signal = np.mean(signal_usage_over_time, axis=0)
avg_reward_per_signal = np.mean(signal_rewards_over_time, axis=0)
signal_usage_array = np.array(signal_usage_over_time)
avg_signal_usage = signal_usage_array.mean(axis=0)

plt.figure()
plt.scatter(final_cost_to_reward_ratio, avg_signal_usage, c=avg_signal_usage, cmap='plasma', edgecolor='k')
plt.xlabel("Signal Cost / Reward Ratio")
plt.ylabel("Average Signal Usage")
plt.title("Signal Usage vs Relative Cost to Reward")
plt.grid(True)
plt.colorbar(label="Usage Intensity")
plt.savefig(os.path.join(output_dir, "signal_cost_to_reward_vs_usage.png"), dpi=300)
plt.show()

plt.figure()
plt.scatter(avg_usage_per_signal, avg_reward_per_signal, color='purple')
plt.xlabel("Average Signal Usage Density")
plt.ylabel("Average Signal Reward")
plt.title("Signal Usage vs Signal Reward")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "signal_usage_vs_reward.png"), dpi=300)
plt.show()

plt.figure()
plt.plot(cooperation_rates_signals, label='Cooperation Rate', color='blue')
plt.plot(defection_rates_signals, label='Defection Rate', color='red')
plt.xlabel("Generation")
plt.ylabel("Rate")
plt.title("Cooperation and Defection Rates in Strategies Over Time")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "cooperation_defection_strategies.png"), dpi=300)
plt.show()

plt.figure()
plt.plot(cooperation_rates, label='Cooperation Rate', color='blue')
plt.plot(defection_rates, label='Defection Rate', color='red')
plt.xlabel("Generation")
plt.ylabel("Rate")
plt.title("Cooperation and Defection Rates Over Time")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "cooperation_defection.png"), dpi=300)
plt.show()

plt.figure()
plt.plot(cc_rates, label='CC', color='green')
plt.plot(cd_rates, label='CD', color='orange')
plt.plot(dd_rates, label='DD', color='red')
plt.xlabel("Generation")
plt.ylabel("State Density")
plt.title("Game States (CC, CD, DD) Over Time")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "state_densities.png"), dpi=300)
plt.show()

plt.figure()
plt.imshow(np.array(signal_usage_over_time).T, cmap='viridis', aspect='auto', origin='lower')
plt.colorbar(label='Usage Probability')
plt.xlabel("Generation")
plt.ylabel("Signal Index")
plt.title("Signal Usage Over Time")
plt.savefig(os.path.join(output_dir, "signal_usage_heatmap.png"), dpi=300)
plt.show()

plt.figure()
plt.plot(coop_avg_rewards, label='Cooperators Avg Reward', color='blue')
plt.plot(defect_avg_rewards, label='Defectors Avg Reward', color='red')
plt.xlabel("Generation")
plt.ylabel("Average Reward")
plt.title("Average Reward by Strategy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "avg_rewards_by_strategy.png"), dpi=300)
plt.show()

plt.figure()
plt.imshow(np.array(coop_strategy_over_time).T, cmap='YlGnBu', aspect='auto', origin='lower')
plt.colorbar(label='C Probability')
plt.xlabel("Generation")
plt.ylabel("Signal Index")
plt.title("Cooperation Strategy Over Time Per Signal")
plt.savefig(os.path.join(output_dir, "coop_strategy_heatmap.png"), dpi=300)
plt.show()

plt.figure()
plt.imshow(np.array(signal_rewards_over_time).T, cmap='magma', aspect='auto', origin='lower')
plt.colorbar(label='Avg Reward')
plt.xlabel("Generation")
plt.ylabel("Signal Index")
plt.title("Signal Average Rewards Over Time")
plt.savefig(os.path.join(output_dir, "signal_rewards_heatmap.png"), dpi=300)
plt.show()

plt.figure()
plt.plot(signal_power_over_time, color='purple')
plt.xlabel("Generation (x100)")
plt.ylabel("Sum of Squared Signal Usage")
plt.title("Signal Usage Concentration Over Time")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "signal_power_over_time.png"), dpi=300)
plt.show()

# --- خلاصه میانگین نرخ‌ها ---
results_summary = {
    "Avg Cooperation Rate": np.mean(cooperation_rates),
    "Avg Defection Rate": np.mean(defection_rates),
    "Avg CC Rate": np.mean(cc_rates),
    "Avg CD Rate": np.mean(cd_rates),
    "Avg DD Rate": np.mean(dd_rates),
}
summary_df = pd.DataFrame([results_summary])

# چاپ در ترمینال
print("\n=== Summary of Average Rates Over Time ===")
print(summary_df.to_string(index=False))

# ذخیره به صورت CSV
csv_path = os.path.join(output_dir, "average_rates_summary.csv")
summary_df.to_csv(csv_path, index=False)
print(f"\nSummary table saved to: {csv_path}")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import softmax
import os
import pandas as pd
from matplotlib.animation import PillowWriter
from joblib import Parallel, delayed
import time

# --- Base path to save all outputs ---
base_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_runs_output")
os.makedirs(base_output_path, exist_ok=True)

# ----------------------------------------------------------------------------------
# | Main simulation function: This function performs a complete run with specific parameters |
# ----------------------------------------------------------------------------------
def run_simulation(params):
    """
    Executes a complete simulation with the given parameters and saves the results.
    """
    # --- Extract parameters from the dictionary ---
    L = params['L']
    n_signals = params['n_signals']
    nu_p = params['nu_p']
    nu_s = params['nu_s']
    rounds = 10000
    beta = params['beta']
    cmax = params['cmax']
    d_sigma = params['d_sigma']
    n_mutation_signals = params['n_mutation_signals']
    samples = 20
    run_name = params['run_name']
    R = params['R'] 
    P = 1
    T = 5
    S = 0
    # --- Create unique output folders for this run ---
    run_dir = os.path.join(base_output_path, run_name)
    output_dir = os.path.join(run_dir, "outputs")
    csv_dir = os.path.join(run_dir, "csv_data")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    print(f"Starting run: {run_name}...")
    start_time = time.time()

    # --- Initialization ---
    signal_costs = np.random.uniform(0, cmax, size=n_signals)
    payoff_matrix = np.array([
        [P, T],
        [S, R]
    ])
    signal_probs = np.random.dirichlet(np.ones(n_signals), size=(L, L))
    signal_response = np.random.randint(0, 2, size=(L, L, n_signals))

    # --- Lists for storing data ---
    cooperation_rates = []
    defection_rates = []
    cooperation_rates_signals = []
    defection_rates_signals = []
    cc_rates = []
    cd_rates = []
    dd_rates = []
    coop_avg_rewards = []
    defect_avg_rewards = []
    signal_usage_over_time = []
    coop_strategy_over_time = []
    signal_rewards_over_time = []
    signal_power_over_time = []
    signal_cost_to_reward_numer = np.zeros(n_signals)
    signal_cost_to_reward_denom = np.zeros(n_signals)
    animation_frames = []

    # --- Main Loop ---
    for gen in range(rounds):
        scores = np.zeros((L, L))
        cc = cd = dd = 0
        total_coop_actions = 0
        total_defect_actions = 0

        signal_usage = np.zeros(n_signals)
        signal_coop_usage = np.zeros(n_signals)
        signal_total_reward = np.zeros(n_signals)
        signal_counts = np.zeros(n_signals)

        coop_score_sum = 0
        defect_score_sum = 0
        coop_count = 0
        defect_count = 0

        for r in range(L):
            for c in range(L):
                neighbors = [((r + 1) % L, c), (r, (c + 1) % L)]
                for nr, nc in neighbors:
                    s1 = np.random.choice(n_signals, p=signal_probs[r, c])
                    s2 = np.random.choice(n_signals, p=signal_probs[nr, nc])
                    a1 = signal_response[r, c, s2]
                    a2 = signal_response[nr, nc, s1]
                    r1, r2 = payoff_matrix[a1, a2], payoff_matrix[a2, a1]

                    scores[r, c] += r1 - signal_costs[s1]
                    scores[nr, nc] += r2 - signal_costs[s2]

                    total_coop_actions += a1 + a2
                    total_defect_actions += 2 - (a1 + a2)
                    if a1 == 1 and a2 == 1: cc += 1
                    elif a1 == 0 and a2 == 0: dd += 1
                    else: cd += 1
                    
                    if a1 == 1:
                        coop_score_sum += r1 - signal_costs[s1]
                        coop_count += 1
                    else:
                        defect_score_sum += r1 - signal_costs[s1]
                        defect_count += 1
                    
                    if a2 == 1:
                        coop_score_sum += r2 - signal_costs[s2]
                        coop_count += 1
                    else:
                        defect_score_sum += r2 - signal_costs[s2]
                        defect_count += 1

                    for s, r_val, a_other in [(s1, r1, a2), (s2, r2, a1)]:
                        signal_usage[s] += 1
                        signal_total_reward[s] += r_val - signal_costs[s]
                        signal_counts[s] += 1
                        signal_coop_usage[s] += a_other
                        if r_val > 0:
                            signal_cost_to_reward_numer[s] += signal_costs[s]
                            signal_cost_to_reward_denom[s] += r_val

        # --- Natural Selection ---
        new_signal_probs = np.empty_like(signal_probs)
        new_signal_response = np.empty_like(signal_response)

        for r in range(L):
            for c in range(L):
                neighbors = [
                    (r, c), ((r-1)%L, c), ((r+1)%L, c),
                    (r, (c-1)%L), (r, (c+1)%L)
                ]
                scores_neighbors = np.array([scores[x, y] for x, y in neighbors])
                probs = softmax(beta * scores_neighbors)
                idx_selected = np.random.choice(len(neighbors), p=probs)
                pr, pc = neighbors[idx_selected]
                new_signal_probs[r, c] = signal_probs[pr, pc]
                new_signal_response[r, c] = signal_response[pr, pc]

        # --- Mutation ---
        mutation_mask_p = np.random.rand(L, L) < nu_p
        mutation_indices_p = np.where(mutation_mask_p)
        for r, c in zip(*mutation_indices_p):
            j = np.random.randint(n_signals)
            new_signal_probs[r, c, j] += d_sigma
            new_signal_probs[r, c] = np.maximum(new_signal_probs[r, c], 0)
            new_signal_probs[r, c] /= new_signal_probs[r, c].sum()

        mutation_mask_s = np.random.rand(L, L) < nu_s
        mutation_indices_s = np.where(mutation_mask_s)
        for r, c in zip(*mutation_indices_s):
            flip_indices = np.random.choice(n_signals, n_mutation_signals, replace=False)
            new_signal_response[r, c, flip_indices] ^= 1

        signal_probs = new_signal_probs
        signal_response = new_signal_response
        
        # --- Save Stats ---
        if gen % samples == 0:
            coop_rate_strategy = signal_response.sum() / (L * L * n_signals)
            cooperation_rates_signals.append(coop_rate_strategy)
            defection_rates_signals.append(1 - coop_rate_strategy)

            total_interactions = L * L * 2
            cooperation_rates.append(total_coop_actions / (2 * total_interactions))
            defection_rates.append(total_defect_actions / (2 * total_interactions))

            cc_rates.append(cc / total_interactions)
            cd_rates.append(cd / total_interactions)
            dd_rates.append(dd / total_interactions)
            
            coop_avg_rewards.append(coop_score_sum / coop_count if coop_count else 0)
            defect_avg_rewards.append(defect_score_sum / defect_count if defect_count else 0)

            su = signal_usage / (signal_usage.sum() + 1e-9)
            signal_usage_over_time.append(su)
            coop_strategy_over_time.append(signal_coop_usage / (signal_counts + 1e-9))
            signal_rewards_over_time.append(signal_total_reward / (signal_counts + 1e-9))
            signal_power_over_time.append(np.sum(su ** 2))
            
            animation_frames.append(np.mean(signal_response, axis=2))
            if gen % (rounds // 100) == 0: 
                print(f"  [{run_name}] Generation {gen}/{rounds}: Coop Rate = {cooperation_rates[-1]:.3f}")
    
    # --- Animation ---
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(animation_frames[0], cmap='viridis', vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cooperation Level")
    title = ax.set_title("Generation 0")

    def update(frame):
        im.set_array(animation_frames[frame])
        title.set_text(f"Generation {frame * samples}")
        return [im, title]

    ani = animation.FuncAnimation(fig, update, frames=len(animation_frames), interval=200, blit=True)
    gif_path = os.path.join(output_dir, "cooperation_heatmap_animation.gif")
    ani.save(gif_path, writer=PillowWriter(fps=5))
    plt.close()

    # --- Plotting and Saving ---
    
    # Heatmaps
    plot_heatmap(np.array(signal_usage_over_time).T, "Signal Usage Over Time", "Generation (x10)", "Signal Index", 'viridis', os.path.join(output_dir, "signal_usage_heatmap.png"), "Usage Probability")
    plot_heatmap(np.array(coop_strategy_over_time).T, "Cooperation Strategy Over Time Per Signal", "Generation (x10)", "Signal Index", 'YlGnBu', os.path.join(output_dir, "coop_strategy_heatmap.png"), "C Probability")
    plot_heatmap(np.array(signal_rewards_over_time).T, "Signal Average Rewards Over Time", "Generation (x10)", "Signal Index", 'magma', os.path.join(output_dir, "signal_rewards_heatmap.png"), "Avg Reward")

    # Scatter and Line Plots
    final_cost_to_reward_ratio = np.divide(signal_cost_to_reward_numer, signal_cost_to_reward_denom + 1e-9)
    avg_usage_per_signal = np.mean(signal_usage_over_time, axis=0)
    avg_reward_per_signal = np.mean(signal_rewards_over_time, axis=0)
    
    save_plot(plt.scatter, {'x': final_cost_to_reward_ratio, 'y': avg_usage_per_signal, 'c': avg_usage_per_signal, 'cmap': 'plasma', 'edgecolor': 'k'}, "Signal Usage vs Relative Cost to Reward", "Signal Cost / Reward Ratio", "Average Signal Usage", os.path.join(output_dir, "signal_cost_to_reward_vs_usage.png"), colorbar_label="Usage Intensity")
    save_plot(plt.scatter, {'x': avg_usage_per_signal, 'y': avg_reward_per_signal, 'color': 'purple'}, "Signal Usage vs Signal Reward", "Average Signal Usage Density", "Average Signal Reward", os.path.join(output_dir, "signal_usage_vs_reward.png"))
    save_plot(plt.plot, [cooperation_rates_signals, defection_rates_signals], "Cooperation and Defection Rates in Strategies", "Generation (x10)", "Rate", os.path.join(output_dir, "cooperation_defection_signals.png"), labels=['Cooperation Rate', 'Defection Rate'], colors=['blue', 'red'])
    save_plot(plt.plot, [cooperation_rates, defection_rates], "Cooperation and Defection Rates in Actions", "Generation (x10)", "Rate", os.path.join(output_dir, "cooperation_defection_actions.png"), labels=['Cooperation Rate', 'Defection Rate'], colors=['blue', 'red'])
    save_plot(plt.plot, [cc_rates, cd_rates, dd_rates], "Game States (CC, CD, DD) Over Time", "Generation (x10)", "State Density", os.path.join(output_dir, "state_densities.png"), labels=['CC', 'CD', 'DD'], colors=['green', 'orange', 'red'])
    save_plot(plt.plot, [coop_avg_rewards, defect_avg_rewards], "Average Reward by Strategy", "Generation (x10)", "Average Reward", os.path.join(output_dir, "avg_rewards_by_strategy.png"), labels=['Cooperators Avg Reward', 'Defectors Avg Reward'], colors=['blue', 'red'])
    save_plot(plt.plot, [signal_power_over_time], "Signal Usage Concentration Over Time", "Generation (x10)", "Sum of Squared Signal Usage", os.path.join(output_dir, "signal_power_over_time.png"), colors=['purple'])

    # --- Save CSVs ---
    pd.DataFrame(signal_usage_over_time).T.to_csv(os.path.join(csv_dir, "signal_usage_over_time.csv"), index=False)
    pd.DataFrame(coop_strategy_over_time).T.to_csv(os.path.join(csv_dir, "coop_strategy_over_time.csv"), index=False)
    pd.DataFrame(signal_rewards_over_time).T.to_csv(os.path.join(csv_dir, "signal_rewards_over_time.csv"), index=False)
    pd.DataFrame({"final_cost_to_reward_ratio": final_cost_to_reward_ratio, "avg_usage_per_signal": avg_usage_per_signal, "avg_reward_per_signal": avg_reward_per_signal}).to_csv(os.path.join(csv_dir, "signal_cost_reward_usage.csv"), index=False)
    pd.DataFrame({"cooperation_rates_signals": cooperation_rates_signals, "defection_rates_signals": defection_rates_signals}).to_csv(os.path.join(csv_dir, "cooperation_defection_signals.csv"), index=False)
    pd.DataFrame({"cooperation_rates": cooperation_rates, "defection_rates": defection_rates}).to_csv(os.path.join(csv_dir, "cooperation_defection_actions.csv"), index=False)
    pd.DataFrame({"cc_rates": cc_rates, "cd_rates": cd_rates, "dd_rates": dd_rates}).to_csv(os.path.join(csv_dir, "state_densities.csv"), index=False)
    pd.DataFrame({"coop_avg_rewards": coop_avg_rewards, "defect_avg_rewards": defect_avg_rewards}).to_csv(os.path.join(csv_dir, "avg_rewards_by_strategy.csv"), index=False)
    pd.DataFrame({"signal_power_over_time": signal_power_over_time}).to_csv(os.path.join(csv_dir, "signal_power_over_time.csv"), index=False)

    end_time = time.time()
    print(f"End of run: {run_name}. Total time: {end_time - start_time:.2f} seconds.")
    return f"Run {run_name} completed successfully."

# --- Helper functions for plotting ---
def plot_heatmap(data, title, xlabel, ylabel, cmap, filename, cbar_label):
    plt.figure(figsize=(10, 6))
    im = plt.imshow(data, aspect='auto', cmap=cmap, origin='lower')
    plt.colorbar(im, label=cbar_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename, dpi=300)
    plt.close()

def save_plot(plot_func, data, title, xlabel, ylabel, filename, labels=None, colors=None, colorbar_label=None):
    plt.figure(figsize=(10, 6))
    if plot_func == plt.plot:
        for i, d in enumerate(data):
            plot_func(d, label=labels[i] if labels else None, color=colors[i] if colors else None)
        if labels:
            plt.legend()
    elif plot_func == plt.scatter:
        handle = plot_func(**data)
        if colorbar_label:
            plt.colorbar(handle, label=colorbar_label)
            
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close()

# ----------------------------------------------------------------------------------
# | Program entry point: Define parameters and execute parallel simulations |
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Define different parameter sets for execution ---
    parameter_sets = [
        # R
        {'L': 40, 'n_signals': 100, 'nu_p': 0.001, 'nu_s': 0.001, 'rounds': 20000, 'beta': 1.0, 'cmax': 0.5, 'd_sigma': 0.2, 'n_mutation_signals': 10, 'R':1.5},
        {'L': 40, 'n_signals': 100, 'nu_p': 0.001, 'nu_s': 0.001, 'rounds': 20000, 'beta': 1.0, 'cmax': 0.5, 'd_sigma': 0.2, 'n_mutation_signals': 10, 'R':2},
        {'L': 40, 'n_signals': 100, 'nu_p': 0.001, 'nu_s': 0.001, 'rounds': 20000, 'beta': 1.0, 'cmax': 0.5, 'd_sigma': 0.2, 'n_mutation_signals': 10, 'R':4},
        {'L': 40, 'n_signals': 100, 'nu_p': 0.001, 'nu_s': 0.001, 'rounds': 20000, 'beta': 1.0, 'cmax': 0.5, 'd_sigma': 0.2, 'n_mutation_signals': 10, 'R':5},
        {'L': 40, 'n_signals': 100, 'nu_p': 0.001, 'nu_s': 0.001, 'rounds': 20000, 'beta': 1.0, 'cmax': 0.5, 'd_sigma': 0.2, 'n_mutation_signals': 10, 'R':6},

        # ns
        {'L': 40, 'n_signals': 100, 'nu_p': 0.001, 'nu_s': 0.001, 'rounds': 20000, 'beta': 1.0, 'cmax': 0.5, 'd_sigma': 0.2, 'n_mutation_signals': 1, 'R':3},
        {'L': 40, 'n_signals': 100, 'nu_p': 0.001, 'nu_s': 0.001, 'rounds': 20000, 'beta': 1.0, 'cmax': 0.5, 'd_sigma': 0.2, 'n_mutation_signals': 5, 'R':3},
        {'L': 40, 'n_signals': 100, 'nu_p': 0.001, 'nu_s': 0.001, 'rounds': 20000, 'beta': 1.0, 'cmax': 0.5, 'd_sigma': 0.2, 'n_mutation_signals': 20, 'R':3},
        {'L': 40, 'n_signals': 100, 'nu_p': 0.001, 'nu_s': 0.001, 'rounds': 20000, 'beta': 1.0, 'cmax': 0.5, 'd_sigma': 0.2, 'n_mutation_signals': 30, 'R':3},
        {'L': 40, 'n_signals': 100, 'nu_p': 0.001, 'nu_s': 0.001, 'rounds': 20000, 'beta': 1.0, 'cmax': 0.5, 'd_sigma': 0.2, 'n_mutation_signals': 50, 'R':3},

        # nu_s
        {'L': 40, 'n_signals': 100, 'nu_p': 0.001, 'nu_s': 0.00001, 'rounds': 20000, 'beta': 1.0, 'cmax': 0.5, 'd_sigma': 0.2, 'n_mutation_signals': 10, 'R':3},
        {'L': 40, 'n_signals': 100, 'nu_p': 0.001, 'nu_s': 0.0001, 'rounds': 20000, 'beta': 1.0, 'cmax': 0.5, 'd_sigma': 0.2, 'n_mutation_signals': 10, 'R':3},
        {'L': 40, 'n_signals': 100, 'nu_p': 0.001, 'nu_s': 0.01, 'rounds': 20000, 'beta': 1.0, 'cmax': 0.5, 'd_sigma': 0.2, 'n_mutation_signals': 10, 'R':3},
        {'L': 40, 'n_signals': 100, 'nu_p': 0.001, 'nu_s': 0.1, 'rounds': 20000, 'beta': 1.0, 'cmax': 0.5, 'd_sigma': 0.2, 'n_mutation_signals': 10, 'R':3},
        {'L': 40, 'n_signals': 100, 'nu_p': 0.001, 'nu_s': 1, 'rounds': 20000, 'beta': 1.0, 'cmax': 0.5, 'd_sigma': 0.2, 'n_mutation_signals': 10, 'R':3},

        # nu_p
        {'L': 40, 'n_signals': 100, 'nu_p': 0.00001, 'nu_s': 0.001, 'rounds': 20000, 'beta': 1.0, 'cmax': 0.5, 'd_sigma': 0.2, 'n_mutation_signals': 10, 'R':3},
        {'L': 40, 'n_signals': 100, 'nu_p': 0.001, 'nu_s': 0.001, 'rounds': 20000, 'beta': 1.0, 'cmax': 0.5, 'd_sigma': 0.2, 'n_mutation_signals': 10, 'R':3},
        {'L': 40, 'n_signals': 100, 'nu_p': 0.01, 'nu_s': 0.001, 'rounds': 20000, 'beta': 1.0, 'cmax': 0.5, 'd_sigma': 0.2, 'n_mutation_signals': 10, 'R':3},
        {'L': 40, 'n_signals': 100, 'nu_p': 0.1, 'nu_s': 0.001, 'rounds': 20000, 'beta': 1.0, 'cmax': 0.5, 'd_sigma': 0.2, 'n_mutation_signals': 10, 'R':3},
        {'L': 40, 'n_signals': 100, 'nu_p': 1, 'nu_s': 0.001, 'rounds': 20000, 'beta': 1.0, 'cmax': 0.5, 'd_sigma': 0.2, 'n_mutation_signals': 10, 'R':3},
    ]

    # --- Add a unique name to each parameter set ---
    for i, params in enumerate(parameter_sets):
        params['run_name'] = f"run_{i+1}_L_{params['L']}_beta_{params['beta']}_cmax_{params['cmax']}_nusignal_{params['nu_p']}_nus_{params['nu_s']}_dsigma{params['d_sigma']}_ns_{params['n_mutation_signals']}_R_{params['R']}_nsignals_{params['n_signals']}"

    n_jobs = -1 
    print(f"Starting parallel execution of {len(parameter_sets)} simulations on {n_jobs if n_jobs != -1 else 'all'} cores...")

    # --- Parallel execution of simulations using Joblib ---
    results = Parallel(n_jobs=n_jobs)(delayed(run_simulation)(params) for params in parameter_sets)

    print("\n" + "="*50)
    print("All simulations finished successfully.")
    print("Results of each run are stored in their specific folder under 'all_runs_output'.")
    print("Results returned from each process:")
    for res in results:
        print(f"- {res}")
    print("="*50)
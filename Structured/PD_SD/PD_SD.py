import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for safe parallel processing
import matplotlib.pyplot as plt
from scipy.special import softmax
import os
import pandas as pd
from joblib import Parallel, delayed
import time

# --- Base path for saving all outputs ---
base_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comprehensive_mixed_game_output")
os.makedirs(base_output_path, exist_ok=True)

# ----------------------------------------------------------------------------------
# | Helper functions for plotting                                                  |
# ----------------------------------------------------------------------------------
def plot_heatmap(data, title, xlabel, ylabel, cmap, filename, cbar_label):
    plt.figure(figsize=(10, 6))
    im = plt.imshow(data, aspect='auto', cmap=cmap, origin='lower')
    plt.colorbar(im, label=cbar_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close('all')

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
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close('all')

# ----------------------------------------------------------------------------------
# | Helper function for vectorized sampling from categorical distributions          |
# ----------------------------------------------------------------------------------
def _sample_categorical(probs):
    """Vectorized sampling from categorical distributions along the last axis (inverse CDF method)."""
    cdf = np.cumsum(probs, axis=-1)
    cdf = cdf / cdf[..., -1:]
    u = np.random.rand(*probs.shape[:-1], 1)
    return (cdf < u).sum(axis=-1)

# ----------------------------------------------------------------------------------
# | Main simulation function                                                       |
# ----------------------------------------------------------------------------------
def run_simulation(params):
    L = params['L']
    n_signals = params['n_signals']
    nu_p = params['nu_p']
    nu_s = params['nu_s']
    rounds = params.get('rounds', 20000)
    beta = params['beta']
    cmax = params['cmax']
    d_sigma = params['d_sigma']
    n_mutation_signals = params['n_mutation_signals']
    samples = 20
    
    p_pd = params['p_pd']
    payoff_PD = params['payoff_PD']
    payoff_SD = params['payoff_SD']
    run_name = f"Run_P_PD_{p_pd:.1f}"

    # --- Create output directories ---
    run_dir = os.path.join(base_output_path, run_name)
    output_dir = os.path.join(run_dir, "plots_and_animations")
    csv_dir = os.path.join(run_dir, "csv_data")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    print(f"Starting run: {run_name}...")
    start_time = time.time()

    # --- Initialization ---
    signal_costs = np.random.uniform(0, cmax, size=n_signals)
    signal_probs = np.random.dirichlet(np.ones(n_signals), size=(L, L))
    signal_response = np.random.randint(0, 2, size=(L, L, n_signals))

    # --- Fixed index grids and neighbor offsets (precomputed for speed) ---
    R, C = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    Rp1 = (R + 1) % L           # Bottom neighbor (r+1, c)
    Cp1 = (C + 1) % L           # Right neighbor (r, c+1)
    sel_dr = np.array([0, -1, 1, 0, 0])   # Offsets for 5 selection options: [self, up, down, left, right]
    sel_dc = np.array([0, 0, 0, -1, 1])

    # --- Lists for storing data ---
    cooperation_rates = []
    defection_rates = []
    cooperation_rates_signals = []
    defection_rates_signals = []
    cc_rates, cd_rates, dd_rates = [], [], []
    coop_avg_rewards, defect_avg_rewards = [], []
    
    signal_usage_over_time = []
    coop_strategy_over_time = []
    signal_rewards_over_time = []
    signal_power_over_time = []
    signal_cost_to_reward_numer = np.zeros(n_signals)
    signal_cost_to_reward_denom = np.zeros(n_signals)
    
    # --- Fitness variables (Benefit - Cost) ---
    signal_benefit_minus_cost_over_time = [] 

    # --- Main Loop ---
    for gen in range(rounds):
        # =========================================================================
        # | Game phase (vectorized): Each edge once, with same logic as main loop |
        # =========================================================================
        # Independent signal sampling for four roles of each cell
        S_focal_down  = _sample_categorical(signal_probs)   # s1 bottom edge
        S_emit_down   = _sample_categorical(signal_probs)   # s2 bottom neighbor
        S_focal_right = _sample_categorical(signal_probs)   # s1 right edge
        S_emit_right  = _sample_categorical(signal_probs)   # s2 right neighbor

        # Random game selection (PD or SD) for each edge
        use_pd_down  = np.random.rand(L, L) < p_pd
        use_pd_right = np.random.rand(L, L) < p_pd

        # --- Bottom edge: (r,c) with (r+1,c) ---
        s1_d = S_focal_down
        s2_d = S_emit_down[Rp1, C]
        a1_d = signal_response[R, C, s2_d]
        a2_d = signal_response[Rp1, C, s1_d]
        r1_d = np.where(use_pd_down, payoff_PD[a1_d, a2_d], payoff_SD[a1_d, a2_d])
        r2_d = np.where(use_pd_down, payoff_PD[a2_d, a1_d], payoff_SD[a2_d, a1_d])
        f1_d = r1_d - signal_costs[s1_d]
        f2_d = r2_d - signal_costs[s2_d]

        # --- Right edge: (r,c) with (r,c+1) ---
        s1_r = S_focal_right
        s2_r = S_emit_right[R, Cp1]
        a1_r = signal_response[R, C, s2_r]
        a2_r = signal_response[R, Cp1, s1_r]
        r1_r = np.where(use_pd_right, payoff_PD[a1_r, a2_r], payoff_SD[a1_r, a2_r])
        r2_r = np.where(use_pd_right, payoff_PD[a2_r, a1_r], payoff_SD[a2_r, a1_r])
        f1_r = r1_r - signal_costs[s1_r]
        f2_r = r2_r - signal_costs[s2_r]

        # --- Income per individual = average of 4 games (2 focal + 2 neighbors) ---
        scores_sum = f1_d + f1_r + np.roll(f2_d, 1, axis=0) + np.roll(f2_r, 1, axis=1)
        scores = scores_sum / 4.0

        # --- Aggregate signal event stats (4 events per cell) ---
        signals_all = np.concatenate([s1_d.ravel(), s2_d.ravel(), s1_r.ravel(), s2_r.ravel()])
        rewards_all = np.concatenate([r1_d.ravel(), r2_d.ravel(), r1_r.ravel(), r2_r.ravel()])
        fitness_all = np.concatenate([f1_d.ravel(), f2_d.ravel(), f1_r.ravel(), f2_r.ravel()])
        partner_all = np.concatenate([a2_d.ravel(), a1_d.ravel(), a2_r.ravel(), a1_r.ravel()]).astype(float)
        actions_all = np.concatenate([a1_d.ravel(), a2_d.ravel(), a1_r.ravel(), a2_r.ravel()])

        signal_usage = np.bincount(signals_all, minlength=n_signals).astype(float)
        signal_counts = signal_usage.copy()
        signal_total_reward = np.bincount(signals_all, weights=rewards_all, minlength=n_signals)
        signal_total_benefit_minus_cost = np.bincount(signals_all, weights=fitness_all, minlength=n_signals)
        signal_coop_usage = np.bincount(signals_all, weights=partner_all, minlength=n_signals)

        pos = rewards_all > 0
        signal_cost_to_reward_numer += np.bincount(signals_all[pos], weights=signal_costs[signals_all[pos]], minlength=n_signals)
        signal_cost_to_reward_denom += np.bincount(signals_all[pos], weights=rewards_all[pos], minlength=n_signals)

        # --- CC/CD/DD counts and actions ---
        cc = int(((a1_d == 1) & (a2_d == 1)).sum() + ((a1_r == 1) & (a2_r == 1)).sum())
        dd = int(((a1_d == 0) & (a2_d == 0)).sum() + ((a1_r == 0) & (a2_r == 0)).sum())
        cd = 2 * L * L - cc - dd

        total_coop_actions = int((actions_all == 1).sum())
        total_defect_actions = int((actions_all == 0).sum())

        coop_mask = actions_all == 1
        coop_score_sum = fitness_all[coop_mask].sum()
        coop_count = int(coop_mask.sum())
        defect_score_sum = fitness_all[~coop_mask].sum()
        defect_count = int((~coop_mask).sum())

        # =========================================================================
        # | Natural selection (vectorized) — based on average income per individual |
        # =========================================================================
        neigh_scores = np.stack([
            scores,
            np.roll(scores, 1, axis=0),
            np.roll(scores, -1, axis=0),
            np.roll(scores, 1, axis=1),
            np.roll(scores, -1, axis=1),
        ], axis=2)

        z = beta * neigh_scores
        z -= z.max(axis=2, keepdims=True)
        e = np.exp(z)
        sel_probs = e / e.sum(axis=2, keepdims=True)

        idx_selected = _sample_categorical(sel_probs)       # (L, L) in range 0..4
        src_r = (R + sel_dr[idx_selected]) % L
        src_c = (C + sel_dc[idx_selected]) % L
        new_signal_probs = signal_probs[src_r, src_c]
        new_signal_response = signal_response[src_r, src_c]

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
            signal_benefit_minus_cost_over_time.append(signal_total_benefit_minus_cost / (signal_counts + 1e-9))
            
            signal_power_over_time.append(np.sum(su ** 2))

        # 🌟 Print progress to console every 100 rounds precisely 🌟
        if (gen + 1) % 100 == 0:
            print(f"  [{run_name}] - Round {gen + 1} out of {rounds} completed.")

    # --- Generate plots (no animation) ---
    # Heatmaps
    plot_heatmap(np.array(signal_usage_over_time).T, "Signal Usage Over Time", "Generation (x10)", "Signal Index", 'viridis', os.path.join(output_dir, "signal_usage_heatmap.png"), "Usage Probability")
    plot_heatmap(np.array(coop_strategy_over_time).T, "Cooperation Strategy Over Time Per Signal", "Generation (x10)", "Signal Index", 'YlGnBu', os.path.join(output_dir, "coop_strategy_heatmap.png"), "C Probability")
    plot_heatmap(np.array(signal_benefit_minus_cost_over_time).T, "Signal Fitness (Benefit - Cost) Over Time", "Generation (x10)", "Signal Index", 'magma', os.path.join(output_dir, "signal_fitness_heatmap.png"), "Benefit - Cost")

    # Line Plots and Scatters
    final_cost_to_reward_ratio = np.divide(signal_cost_to_reward_numer, signal_cost_to_reward_denom + 1e-9)
    avg_usage_per_signal = np.mean(signal_usage_over_time, axis=0)
    avg_fitness_per_signal = np.mean(signal_benefit_minus_cost_over_time, axis=0) 
    
    # Signal scatter plots
    save_plot(plt.scatter, {'x': final_cost_to_reward_ratio, 'y': avg_usage_per_signal, 'c': avg_usage_per_signal, 'cmap': 'plasma', 'edgecolor': 'k'}, "Signal Usage vs Relative Cost to Reward", "Signal Cost / Reward Ratio", "Average Signal Usage", os.path.join(output_dir, "signal_cost_to_reward_vs_usage.png"), colorbar_label="Usage Intensity")
    save_plot(plt.scatter, {'x': avg_fitness_per_signal, 'y': avg_usage_per_signal, 'color': 'darkgreen', 'alpha': 0.7, 'edgecolor': 'black'}, "Signal Usage vs True Fitness (Benefit - Cost)", "Average Signal Fitness (Benefit - Cost)", "Average Signal Usage Density", os.path.join(output_dir, "signal_usage_vs_fitness.png"))
    
    save_plot(plt.scatter, {'x': signal_costs, 'y': avg_usage_per_signal, 'color': 'teal', 'alpha': 0.7, 'edgecolor': 'black'}, "Signal Usage vs Signal Cost", "Signal Cost", "Average Signal Usage", os.path.join(output_dir, "signal_usage_vs_cost.png"))
    
    save_plot(plt.plot, [cooperation_rates, defection_rates], "Cooperation and Defection Rates", "Generation (x10)", "Rate", os.path.join(output_dir, "cooperation_defection_actions.png"), labels=['Cooperation Rate', 'Defection Rate'], colors=['blue', 'red'])
    save_plot(plt.plot, [cc_rates, cd_rates, dd_rates], "Game States (CC, CD, DD) Over Time", "Generation (x10)", "State Density", os.path.join(output_dir, "state_densities.png"), labels=['CC', 'CD', 'DD'], colors=['green', 'orange', 'red'])
    save_plot(plt.plot, [coop_avg_rewards, defect_avg_rewards], "Average Fitness by Strategy", "Generation (x10)", "Average Fitness (Benefit-Cost)", os.path.join(output_dir, "avg_fitness_by_strategy.png"), labels=['Cooperators', 'Defectors'], colors=['blue', 'red'])

    # --- Save CSVs ---
    pd.DataFrame(signal_usage_over_time).T.to_csv(os.path.join(csv_dir, "signal_usage_over_time.csv"), index=False)
    pd.DataFrame(coop_strategy_over_time).T.to_csv(os.path.join(csv_dir, "coop_strategy_over_time.csv"), index=False)
    pd.DataFrame(signal_benefit_minus_cost_over_time).T.to_csv(os.path.join(csv_dir, "signal_benefit_minus_cost_over_time.csv"), index=False)
    
    pd.DataFrame({
        "avg_fitness_benefit_minus_cost": avg_fitness_per_signal, 
        "avg_usage_per_signal": avg_usage_per_signal,
        "final_cost_to_reward_ratio": final_cost_to_reward_ratio
    }).to_csv(os.path.join(csv_dir, "signal_usage_vs_fitness_data.csv"), index=False)
    
    pd.DataFrame({
        "signal_cost": signal_costs,
        "avg_usage_per_signal": avg_usage_per_signal
    }).to_csv(os.path.join(csv_dir, "signal_usage_vs_cost_data.csv"), index=False)
    
    pd.DataFrame({"final_cost_to_reward_ratio": final_cost_to_reward_ratio, "avg_usage_per_signal": avg_usage_per_signal, "avg_fitness_per_signal": avg_fitness_per_signal}).to_csv(os.path.join(csv_dir, "signal_cost_fitness_usage.csv"), index=False)
    pd.DataFrame({"cooperation_rates": cooperation_rates, "defection_rates": defection_rates}).to_csv(os.path.join(csv_dir, "cooperation_defection_actions.csv"), index=False)
    pd.DataFrame({"cc_rates": cc_rates, "cd_rates": cd_rates, "dd_rates": dd_rates}).to_csv(os.path.join(csv_dir, "state_densities.csv"), index=False)
    pd.DataFrame({"coop_avg_fitness": coop_avg_rewards, "defect_avg_fitness": defect_avg_rewards}).to_csv(os.path.join(csv_dir, "avg_fitness_by_strategy.csv"), index=False)

    # --- Extract steady-state data for final plot ---
    tail_len = max(1, len(cooperation_rates) // 5) # Last 20%
    steady_state = {
        'P_PD': p_pd,
        'C': np.mean(cooperation_rates[-tail_len:]),
        'D': np.mean(defection_rates[-tail_len:]),
        'CC': np.mean(cc_rates[-tail_len:]),
        'CD': np.mean(cd_rates[-tail_len:]),
        'DD': np.mean(dd_rates[-tail_len:])
    }

    end_time = time.time()
    print(f"Run finished: {run_name}. Time: {end_time - start_time:.2f} seconds.")
    return steady_state

# ----------------------------------------------------------------------------------
# | Transition graph plotting function                                             |
# ----------------------------------------------------------------------------------
def plot_transition_graph(df, save_path):
    plt.figure(figsize=(7, 5))
    
    plt.plot(df['P_PD'], df['C'], label='C', marker='o', color='blue', linestyle='-', linewidth=1)
    plt.plot(df['P_PD'], df['D'], label='D', marker='o', color='red', linestyle='-', linewidth=1)
    plt.plot(df['P_PD'], df['CC'], label='CC', marker='^', color='gray', linestyle='--', linewidth=1, markersize=5)
    plt.plot(df['P_PD'], df['CD'], label='CD', marker='^', color='darkkhaki', linestyle='--', linewidth=1, markersize=5)
    plt.plot(df['P_PD'], df['DD'], label='DD', marker='^', color='navy', linestyle='--', linewidth=1, markersize=5)

    plt.title('PD-SD Transition')
    plt.xlabel('$P_{PD}$')
    plt.ylabel('Frequency')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(0.05, 0.5), frameon=True, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "PD_SD_Transition_Plot.png"), dpi=300)
    plt.close('all')

# ----------------------------------------------------------------------------------
# | Program entry point                                                            |
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # Prisoner's Dilemma (PD): R=3, S=0, T=5, P=1
    matrix_PD = np.array([
        [1.0, 5.0], 
        [0.0, 3.0]  
    ])
    
    
    matrix_SD = np.array([
        [0, 5.0], 
        [1, 3.0]
    ])
    
    base_params = {
        'L': 40, 
        'n_signals': 100, 
        'nu_p': 0.001, 
        'nu_s': 0.001, 
        'beta': 1.0, 
        'cmax': 0.5, 
        'd_sigma': 0.2, 
        'n_mutation_signals': 10,
        'rounds': 20000, 
        'payoff_PD': matrix_PD,
        'payoff_SD': matrix_SD
    }

    # Generate P_PD probabilities from 0 to 1
    p_pd_values = np.linspace(0, 1.0, 11)
    
    parameter_sets = []
    for p in p_pd_values:
        param_copy = base_params.copy()
        param_copy['p_pd'] = p
        parameter_sets.append(param_copy)

    n_jobs = -1 
    print(f"Starting parallel execution for {len(parameter_sets)} different P_PD values on {n_jobs if n_jobs != -1 else 'all'} cores...")

    # Parallel execution
    results = Parallel(n_jobs=n_jobs)(delayed(run_simulation)(params) for params in parameter_sets)

    print("\n" + "="*50)
    print("All simulations finished. Aggregating data...")
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='P_PD').reset_index(drop=True)
    
    # Save transition data
    csv_path = os.path.join(base_output_path, "transition_data.csv")
    df_results.to_csv(csv_path, index=False)
    
    # Plot final graph
    plot_transition_graph(df_results, base_output_path)
    
    print("Outputs (transition plot + individual plots and CSVs) successfully generated.")
    print("="*50)
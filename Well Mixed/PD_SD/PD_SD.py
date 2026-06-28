import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import softmax
import os
import pandas as pd
from joblib import Parallel, delayed
import time

# --- Base output directory ---
base_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "well_mixed_comprehensive_output")
os.makedirs(base_output_path, exist_ok=True)

# ----------------------------------------------------------------------------------
# | Plotting Helper Functions                                                      |
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

def safe_divide(numer, denom):
    return numer / (denom + 1e-9)

# ----------------------------------------------------------------------------------
# | Main Simulation Function (Well-Mixed)                                          |
# ----------------------------------------------------------------------------------
def run_simulation(params):
    N = params['N']
    n_signals = params['n_signals']
    nu_p = params['nu_p']
    nu_s = params['nu_s']
    rounds = params['rounds']
    beta = params['beta']
    cmax = params['cmax']
    d_sigma = params['d_sigma']
    n_mutation_signals = params['n_mutation_signals']
    save_interval = 20
    
    p_pd = params['p_pd']
    payoff_PD = params['payoff_PD']
    payoff_SD = params['payoff_SD']
    run_name = f"Run_P_PD_{p_pd:.1f}"

    # Directory setup
    run_dir = os.path.join(base_output_path, run_name)
    output_dir = os.path.join(run_dir, "plots")
    csv_dir = os.path.join(run_dir, "csv_data")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    start_time = time.time()

    # Initialization (Signals are sorted by cost)
    signal_costs = np.random.uniform(0, cmax, size=n_signals)
    signal_costs = np.sort(signal_costs)

    signal_probs = np.random.dirichlet(np.ones(n_signals), size=N)
    signal_response = np.random.randint(0, 2, size=(N, n_signals))

    # Data tracking lists
    time_steps = []
    cooperation_rates, defection_rates = [], []
    cooperation_rates_signals, defection_rates_signals = [], []
    cc_rates, cd_rates, dd_rates = [], [], []
    coop_avg_rewards, defect_avg_rewards = [], []
    
    signal_usage_over_time = []
    coop_strategy_over_time = []
    signal_power_over_time = []
    signal_benefit_minus_cost_over_time = [] 
    avg_signal_cost_over_time = []
    
    signal_cost_to_reward_numer = np.zeros(n_signals)
    signal_cost_to_reward_denom = np.zeros(n_signals)

    # Main Evolutionary Loop
    for gen in range(rounds):
        scores = np.zeros(N)
        cc = cd = dd = 0
        coop_players, defect_players = [], []

        signal_usage = np.zeros(n_signals)
        signal_coop_usage = np.zeros(n_signals)
        signal_total_benefit_minus_cost = np.zeros(n_signals) 
        signal_counts = np.zeros(n_signals)

        # Random pairing for well-mixed population
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

        # Vectorized game environment selection
        num_pairs = N // 2
        is_pd = np.random.rand(num_pairs) < p_pd
        
        r1 = np.where(is_pd, payoff_PD[a1, a2], payoff_SD[a1, a2])
        r2 = np.where(is_pd, payoff_PD[a2, a1], payoff_SD[a2, a1])

        cost1 = signal_costs[s1]
        cost2 = signal_costs[s2]

        # Fitness calculation (Benefit - Cost)
        fitness1 = r1 - cost1
        fitness2 = r2 - cost2

        scores[idx1] += fitness1
        scores[idx2] += fitness2

        # Record signal statistics
        for s, gross_r, fit_val, a_other in zip(s1, r1, fitness1, a2):
            signal_usage[s] += 1
            signal_total_benefit_minus_cost[s] += fit_val
            signal_counts[s] += 1
            signal_coop_usage[s] += a_other
            if gross_r > 0:
                signal_cost_to_reward_numer[s] += signal_costs[s]
                signal_cost_to_reward_denom[s] += gross_r

        for s, gross_r, fit_val, a_other in zip(s2, r2, fitness2, a1):
            signal_usage[s] += 1
            signal_total_benefit_minus_cost[s] += fit_val
            signal_counts[s] += 1
            signal_coop_usage[s] += a_other
            if gross_r > 0:
                signal_cost_to_reward_numer[s] += signal_costs[s]
                signal_cost_to_reward_denom[s] += gross_r

        # Action statistics
        total_coop_actions = a1.sum() + a2.sum()
        total_defect_actions = 2 * len(a1) - total_coop_actions

        cc += np.sum((a1 == 1) & (a2 == 1))
        cd += np.sum((a1 != a2))
        dd += np.sum((a1 == 0) & (a2 == 0))

        coop_players.extend(idx1[a1 == 1])
        coop_players.extend(idx2[a2 == 1])
        defect_players.extend(idx1[a1 == 0])
        defect_players.extend(idx2[a2 == 0])

        # Natural Selection (Wright-Fisher)
        exp_scores = np.exp(beta * scores - np.max(beta * scores))
        probs = exp_scores / exp_scores.sum()
        parent_indices = np.random.choice(N, size=N, p=probs)
        new_signal_probs = signal_probs[parent_indices].copy()
        new_signal_response = signal_response[parent_indices].copy()

        # Mutation
        mutation_mask_p = np.random.rand(N) < nu_p
        mut_idxs = np.where(mutation_mask_p)[0]
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
        
        # Save Stats periodically
        if gen % save_interval == 0:
            time_steps.append(gen)
            
            coop_rate_strategy = signal_response.sum() / (N * n_signals)
            cooperation_rates_signals.append(coop_rate_strategy)
            defection_rates_signals.append(1 - coop_rate_strategy)

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

            su = signal_usage / (signal_usage.sum() + 1e-9)
            signal_usage_over_time.append(su)
            coop_strategy_over_time.append(safe_divide(signal_coop_usage, signal_counts))
            signal_benefit_minus_cost_over_time.append(safe_divide(signal_total_benefit_minus_cost, signal_counts))
            
            signal_power_over_time.append(np.sum(su ** 2))
            
            # Record average signal cost paid in this generation
            avg_cost = (np.sum(cost1) + np.sum(cost2)) / N
            avg_signal_cost_over_time.append(avg_cost)

        if (gen + 1) % 10000 == 0:
            print(f"  [{run_name}] - Generation {gen + 1}/{rounds} completed.")

    # --- Plotting ---
    final_cost_to_reward_ratio = safe_divide(signal_cost_to_reward_numer, signal_cost_to_reward_denom)
    avg_usage_per_signal = np.mean(signal_usage_over_time, axis=0)
    avg_fitness_per_signal = np.mean(signal_benefit_minus_cost_over_time, axis=0) 
    
    plot_heatmap(np.array(signal_usage_over_time).T, "Signal Usage Over Time", "Generation (x200)", "Signal Index", 'viridis', os.path.join(output_dir, "signal_usage_heatmap.png"), "Usage Probability")
    plot_heatmap(np.array(coop_strategy_over_time).T, "Cooperation Strategy Over Time Per Signal", "Generation (x200)", "Signal Index", 'YlGnBu', os.path.join(output_dir, "coop_strategy_heatmap.png"), "C Probability")
    plot_heatmap(np.array(signal_benefit_minus_cost_over_time).T, "Signal Fitness (Benefit - Cost) Over Time", "Generation (x200)", "Signal Index", 'magma', os.path.join(output_dir, "signal_fitness_heatmap.png"), "Benefit - Cost")

    save_plot(plt.scatter, {'x': final_cost_to_reward_ratio, 'y': avg_usage_per_signal, 'c': avg_usage_per_signal, 'cmap': 'plasma', 'edgecolor': 'k'}, "Signal Usage vs Relative Cost to Reward", "Signal Cost / Reward Ratio", "Average Signal Usage", os.path.join(output_dir, "signal_cost_to_reward_vs_usage.png"), colorbar_label="Usage Intensity")
    save_plot(plt.scatter, {'x': avg_fitness_per_signal, 'y': avg_usage_per_signal, 'color': 'darkgreen', 'alpha': 0.7, 'edgecolor': 'black'}, "Signal Usage vs True Fitness", "Average Signal Fitness (Benefit - Cost)", "Average Signal Usage Density", os.path.join(output_dir, "signal_usage_vs_fitness.png"))
    save_plot(plt.scatter, {'x': signal_costs, 'y': avg_usage_per_signal, 'color': 'teal', 'alpha': 0.7, 'edgecolor': 'black'}, "Signal Usage vs Signal Cost", "Signal Cost", "Average Signal Usage", os.path.join(output_dir, "signal_usage_vs_cost.png"))
    
    save_plot(plt.plot, [cooperation_rates, defection_rates], "Cooperation and Defection Rates", "Generation (x200)", "Rate", os.path.join(output_dir, "cooperation_defection_actions.png"), labels=['Cooperation Rate', 'Defection Rate'], colors=['blue', 'red'])
    save_plot(plt.plot, [cc_rates, cd_rates, dd_rates], "Game States (CC, CD, DD) Over Time", "Generation (x200)", "State Density", os.path.join(output_dir, "state_densities.png"), labels=['CC', 'CD', 'DD'], colors=['green', 'orange', 'red'])
    save_plot(plt.plot, [coop_avg_rewards, defect_avg_rewards], "Average Fitness by Strategy", "Generation (x200)", "Average Fitness", os.path.join(output_dir, "avg_fitness_by_strategy.png"), labels=['Cooperators', 'Defectors'], colors=['blue', 'red'])
    save_plot(plt.plot, [avg_signal_cost_over_time], "Average Signal Cost Paid Over Time", "Generation (x200)", "Average Cost", os.path.join(output_dir, "avg_signal_cost_over_time.png"), labels=['Avg Cost'], colors=['purple'])

    # --- Save CSVs ---
    pd.DataFrame({"t": time_steps, "cooperation_rates": cooperation_rates, "defection_rates": defection_rates}).to_csv(os.path.join(csv_dir, "cooperation_defection_actions.csv"), index=False)
    pd.DataFrame({"t": time_steps, "cc_rates": cc_rates, "cd_rates": cd_rates, "dd_rates": dd_rates}).to_csv(os.path.join(csv_dir, "state_densities.csv"), index=False)
    pd.DataFrame({"t": time_steps, "coop_avg_fitness": coop_avg_rewards, "defect_avg_fitness": defect_avg_rewards}).to_csv(os.path.join(csv_dir, "avg_fitness_by_strategy.csv"), index=False)
    pd.DataFrame({"t": time_steps, "Avg_Signal_Cost": avg_signal_cost_over_time}).to_csv(os.path.join(csv_dir, "avg_signal_cost_over_time.csv"), index=False)
    
    pd.DataFrame(signal_usage_over_time).T.to_csv(os.path.join(csv_dir, "signal_usage_over_time.csv"), index=False)
    pd.DataFrame(coop_strategy_over_time).T.to_csv(os.path.join(csv_dir, "coop_strategy_over_time.csv"), index=False)
    pd.DataFrame(signal_benefit_minus_cost_over_time).T.to_csv(os.path.join(csv_dir, "signal_benefit_minus_cost_over_time.csv"), index=False)
    
    pd.DataFrame({
        "signal_cost": signal_costs,
        "avg_usage_per_signal": avg_usage_per_signal
    }).to_csv(os.path.join(csv_dir, "signal_usage_vs_cost_data.csv"), index=False)
    
    # Extract steady state data
    tail_len = max(1, len(cooperation_rates) // 5) 
    steady_state = {
        'P_PD': p_pd,
        'C': np.mean(cooperation_rates[-tail_len:]),
        'D': np.mean(defection_rates[-tail_len:]),
        'CC': np.mean(cc_rates[-tail_len:]),
        'CD': np.mean(cd_rates[-tail_len:]),
        'DD': np.mean(dd_rates[-tail_len:])
    }

    print(f"end of simulation: {run_name}. time: {time.time() - start_time:.2f} s.")
    return steady_state

# ----------------------------------------------------------------------------------
# | Transition Graph Plotting Function                                             |
# ----------------------------------------------------------------------------------
def plot_transition_graph(df, save_path):
    plt.figure(figsize=(7, 5))
    
    plt.plot(df['P_PD'], df['C'], label='C', marker='o', color='blue', linestyle='-', linewidth=1)
    plt.plot(df['P_PD'], df['D'], label='D', marker='o', color='red', linestyle='-', linewidth=1)
    plt.plot(df['P_PD'], df['CC'], label='CC', marker='^', color='gray', linestyle='--', linewidth=1, markersize=5)
    plt.plot(df['P_PD'], df['CD'], label='CD', marker='^', color='darkkhaki', linestyle='--', linewidth=1, markersize=5)
    plt.plot(df['P_PD'], df['DD'], label='DD', marker='^', color='navy', linestyle='--', linewidth=1, markersize=5)

    plt.title('PD-SD Transition (Well-Mixed)')
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
# | Entry Point                                                                    |
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # Prisoner's Dilemma (PD): R=3, S=0, T=5, P=1
    matrix_PD = np.array([
        [1.0, 5.0], 
        [0.0, 3.0]  
    ])
    
    # Snowdrift (SD): R=3, S=1, T=5, P=0
    matrix_SD = np.array([
        [0.0, 5.0], 
        [1.0, 3.0]
    ])
    
    base_params = {
        'N': 1800, 
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

    # Generate P_PD values from 0 to 1
    p_pd_values = np.linspace(0, 1.0, 11)
    
    parameter_sets = []
    for p in p_pd_values:
        param_copy = base_params.copy()
        param_copy['p_pd'] = p
        parameter_sets.append(param_copy)

    n_jobs = -1 
    print(f"Starting parallel execution for {len(parameter_sets)} different P_PD values...")

    # Parallel Execution
    results = Parallel(n_jobs=n_jobs)(delayed(run_simulation)(params) for params in parameter_sets)

    print("\n" + "="*50)
    print("All simulations have finished. Aggregating data...")
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='P_PD').reset_index(drop=True)
    
    # Save Transition Data
    csv_path = os.path.join(base_output_path, "transition_data.csv")
    df_results.to_csv(csv_path, index=False)
    
    # Plot Final Transition Graph
    plot_transition_graph(df_results, base_output_path)
    
    print("Outputs (transition diagram + plots and individual CSV files) were generated successfully.")
    print("="*50)
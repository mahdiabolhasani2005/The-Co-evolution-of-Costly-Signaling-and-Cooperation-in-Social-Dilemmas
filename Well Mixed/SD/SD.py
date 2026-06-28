import numpy as np
import matplotlib
# Configures the non-interactive backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import softmax
import os
import pandas as pd
from joblib import Parallel, delayed
import time

# Creates the base output directory.
base_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "well_mixed_parameter_sweep")
os.makedirs(base_output_path, exist_ok=True)

# Generates a heatmap plot.
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

# Saves a generic plot.
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

# Calculates safe division.
def safe_divide(numer, denom):
    return numer / (denom + 1e-9)

# Runs a single simulation execution.
def run_simulation(params):
    # Extracts parameters.
    N = params['N']
    n_signals = params['n_signals']
    nu_p = params['nu_p']
    nu_s = params['nu_s']
    rounds = params['rounds']
    beta = params['beta']
    cmax = params['cmax']
    d_sigma = params['d_sigma']
    n_mutation_signals = params['n_mutation_signals']
    payoff_matrix = params['payoff_matrix']
    
    # Sets the save interval.
    save_interval = 20
    
    run_name = f"Beta_{beta}_nuP_{nu_p}"

    # Creates execution directories.
    run_dir = os.path.join(base_output_path, run_name)
    images_dir = os.path.join(run_dir, "images")
    csv_dir = os.path.join(run_dir, "csv_exports")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    start_time = time.time()

    # Initializes and sorts signal costs.
    signal_costs = np.random.uniform(0, cmax, size=n_signals)
    signal_costs = np.sort(signal_costs)

    # Initializes population arrays.
    signal_probs = np.random.dirichlet(np.ones(n_signals), size=N)
    signal_response = np.random.randint(0, 2, size=(N, n_signals))

    # Initializes statistical tracking lists.
    time_steps = []
    cooperation_rates_signals, defection_rates_signals = [], []
    cooperation_rates, defection_rates = [], []
    cc_rates, cd_rates, dd_rates = [], [], []
    coop_avg_rewards, defect_avg_rewards = [], []
    
    signal_usage_over_time = []
    coop_strategy_over_time = []
    signal_fitness_over_time = [] 
    signal_power_over_time = []
    avg_signal_cost_over_time = []
    
    signal_cost_to_reward_numer = np.zeros(n_signals)
    signal_cost_to_reward_denom = np.zeros(n_signals)

    # Executes the evolutionary loop.
    for gen in range(rounds):
        # Prints execution progress.
        if (gen + 1) % 500 == 0:
            print(f"[{run_name}] Progress: Round {gen + 1} of {rounds} completed.")
            
        # Resets generation metrics.
        scores = np.zeros(N)
        cc = cd = dd = 0
        coop_players, defect_players = [], []

        signal_usage = np.zeros(n_signals)
        signal_coop_usage = np.zeros(n_signals)
        signal_total_fitness = np.zeros(n_signals) 
        signal_counts = np.zeros(n_signals)

        # Pairs individuals randomly.
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

        # Computes true fitness.
        fitness1 = r1 - cost1
        fitness2 = r2 - cost2

        # Updates scores.
        scores[idx1] += fitness1
        scores[idx2] += fitness2

        # Records signal statistics for the first group.
        for s, gross_r, fit_val, a_other in zip(s1, r1, fitness1, a2):
            signal_usage[s] += 1
            signal_total_fitness[s] += fit_val
            signal_counts[s] += 1
            signal_coop_usage[s] += a_other
            if gross_r > 0:
                signal_cost_to_reward_numer[s] += signal_costs[s]
                signal_cost_to_reward_denom[s] += gross_r

        # Records signal statistics for the second group.
        for s, gross_r, fit_val, a_other in zip(s2, r2, fitness2, a1):
            signal_usage[s] += 1
            signal_total_fitness[s] += fit_val
            signal_counts[s] += 1
            signal_coop_usage[s] += a_other
            if gross_r > 0:
                signal_cost_to_reward_numer[s] += signal_costs[s]
                signal_cost_to_reward_denom[s] += gross_r

        # Computes generation totals.
        total_coop_actions = a1.sum() + a2.sum()
        total_defect_actions = 2 * len(a1) - total_coop_actions

        cc += np.sum((a1 == 1) & (a2 == 1))
        cd += np.sum((a1 != a2))
        dd += np.sum((a1 == 0) & (a2 == 0))

        coop_players.extend(idx1[a1 == 1])
        coop_players.extend(idx2[a2 == 1])
        defect_players.extend(idx1[a1 == 0])
        defect_players.extend(idx2[a2 == 0])

        # Executes natural selection.
        exp_scores = np.exp(beta * scores - np.max(beta * scores))
        probs = exp_scores / exp_scores.sum()
        parent_indices = np.random.choice(N, size=N, p=probs)
        new_signal_probs = signal_probs[parent_indices].copy()
        new_signal_response = signal_response[parent_indices].copy()

        # Applies mutations to probabilities and responses.
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

        # Saves statistics periodically.
        if gen % save_interval == 0:
            time_steps.append(gen)
            
            coop_rate1 = signal_response.sum() / (N * n_signals)
            cooperation_rates_signals.append(coop_rate1)
            defection_rates_signals.append(1 - coop_rate1)

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
            signal_fitness_over_time.append(safe_divide(signal_total_fitness, signal_counts))

            norm_usage = signal_usage / N
            signal_power = np.sum(norm_usage ** 2)
            signal_power_over_time.append(signal_power)
            
            # Computes the average signal cost.
            avg_cost = (np.sum(cost1) + np.sum(cost2)) / N
            avg_signal_cost_over_time.append(avg_cost)

    # Generates output plots.
    final_cost_to_reward_ratio = safe_divide(signal_cost_to_reward_numer, signal_cost_to_reward_denom)
    avg_usage_per_signal = np.mean(signal_usage_over_time, axis=0)
    avg_fitness_per_signal = np.mean(signal_fitness_over_time, axis=0)

    plot_heatmap(np.array(signal_usage_over_time).T, "Signal Usage Over Time", "Generation (x200)", "Signal Index", 'viridis', os.path.join(images_dir, "signal_usage_heatmap.png"), "Usage Probability")
    plot_heatmap(np.array(coop_strategy_over_time).T, "Cooperation Strategy Over Time Per Signal", "Generation (x200)", "Signal Index", 'YlGnBu', os.path.join(images_dir, "coop_strategy_heatmap.png"), "C Probability")
    plot_heatmap(np.array(signal_fitness_over_time).T, "Signal Fitness (Benefit - Cost) Over Time", "Generation (x200)", "Signal Index", 'magma', os.path.join(images_dir, "signal_fitness_heatmap.png"), "Benefit - Cost")

    save_plot(plt.scatter, {'x': final_cost_to_reward_ratio, 'y': avg_usage_per_signal, 'c': avg_usage_per_signal, 'cmap': 'plasma', 'edgecolor': 'k'}, "Signal Usage vs Relative Cost to Reward", "Signal Cost / Gross Reward Ratio", "Average Signal Usage", os.path.join(images_dir, "signal_cost_to_reward_vs_usage.png"), colorbar_label="Usage Intensity")
    save_plot(plt.scatter, {'x': avg_fitness_per_signal, 'y': avg_usage_per_signal, 'color': 'darkgreen', 'alpha': 0.7, 'edgecolor': 'black'}, "Signal Usage vs True Fitness (Benefit - Cost)", "Average Signal Fitness (Benefit - Cost)", "Average Signal Usage Density", os.path.join(images_dir, "signal_usage_vs_fitness.png"))
    save_plot(plt.scatter, {'x': signal_costs, 'y': avg_usage_per_signal, 'color': 'teal', 'alpha': 0.7, 'edgecolor': 'black'}, "Signal Usage vs Signal Cost", "Signal Cost", "Average Signal Usage", os.path.join(images_dir, "signal_usage_vs_cost.png"))

    save_plot(plt.plot, [cooperation_rates_signals, defection_rates_signals], "Cooperation and Defection Rates in Strategies", "Generation", "Rate", os.path.join(images_dir, "cooperation_defection_strategies.png"), labels=['Cooperation', 'Defection'], colors=['blue', 'red'])
    save_plot(plt.plot, [cooperation_rates, defection_rates], "Cooperation and Defection Rates (Actions)", "Generation", "Rate", os.path.join(images_dir, "cooperation_defection_actions.png"), labels=['Cooperation', 'Defection'], colors=['blue', 'red'])
    save_plot(plt.plot, [cc_rates, cd_rates, dd_rates], "Game States (CC, CD, DD) Over Time", "Generation", "State Density", os.path.join(images_dir, "state_densities.png"), labels=['CC', 'CD', 'DD'], colors=['green', 'orange', 'red'])
    save_plot(plt.plot, [coop_avg_rewards, defect_avg_rewards], "Average Fitness by Strategy", "Generation", "Average Fitness", os.path.join(images_dir, "avg_rewards_by_strategy.png"), labels=['Cooperators', 'Defectors'], colors=['blue', 'red'])
    save_plot(plt.plot, [signal_power_over_time], "Signal Usage Concentration Over Time", "Generation", "Sum of Squared Signal Usage", os.path.join(images_dir, "signal_power_over_time.png"), colors=['purple'])
    save_plot(plt.plot, [avg_signal_cost_over_time], "Average Signal Cost Paid Over Time", "Generation", "Average Cost Paid", os.path.join(images_dir, "avg_signal_cost_over_time.png"), labels=['Avg Cost'], colors=['purple'])

    # Exports data to CSV formats.
    pd.DataFrame({"t": time_steps, "Cooperation_Rate_Strategies": cooperation_rates_signals, "Defection_Rate_Strategies": defection_rates_signals}).to_csv(os.path.join(csv_dir, "coop_defect_strategies.csv"), index=False)
    pd.DataFrame({"t": time_steps, "Cooperation_Rate": cooperation_rates, "Defection_Rate": defection_rates}).to_csv(os.path.join(csv_dir, "cooperation_defection_actions.csv"), index=False)
    pd.DataFrame({"t": time_steps, "CC_Rate": cc_rates, "CD_Rate": cd_rates, "DD_Rate": dd_rates}).to_csv(os.path.join(csv_dir, "state_densities.csv"), index=False)
    pd.DataFrame({"t": time_steps, "Cooperators_Reward": coop_avg_rewards, "Defectors_Reward": defect_avg_rewards}).to_csv(os.path.join(csv_dir, "avg_rewards_by_strategy.csv"), index=False)
    pd.DataFrame({"t": time_steps, "Signal_Power": signal_power_over_time}).to_csv(os.path.join(csv_dir, "signal_power.csv"), index=False)
    pd.DataFrame({"t": time_steps, "Avg_Signal_Cost": avg_signal_cost_over_time}).to_csv(os.path.join(csv_dir, "avg_signal_cost_over_time.csv"), index=False)
    
    pd.DataFrame(signal_usage_over_time, columns=[f"Sig_{i}" for i in range(n_signals)]).to_csv(os.path.join(csv_dir, "signal_usage_over_time.csv"), index=False)
    pd.DataFrame(coop_strategy_over_time, columns=[f"Sig_{i}" for i in range(n_signals)]).to_csv(os.path.join(csv_dir, "coop_strategy_over_time.csv"), index=False)
    pd.DataFrame(signal_fitness_over_time, columns=[f"Sig_{i}" for i in range(n_signals)]).to_csv(os.path.join(csv_dir, "signal_fitness_over_time.csv"), index=False)

    pd.DataFrame({
        "signal_cost": signal_costs,
        "avg_fitness_benefit_minus_cost": avg_fitness_per_signal, 
        "avg_usage_per_signal": avg_usage_per_signal,
        "final_cost_to_reward_ratio": final_cost_to_reward_ratio
    }).to_csv(os.path.join(csv_dir, "signal_metrics_summary.csv"), index=False)

    # Calculates the steady state metrics.
    tail = max(1, (rounds // save_interval) // 5)
    steady_state = {
        'Beta': beta,
        'nu_p': nu_p,
        'Avg_C_Rate': np.mean(cooperation_rates[-tail:]),
        'Avg_D_Rate': np.mean(defection_rates[-tail:]),
        'Avg_CC': np.mean(cc_rates[-tail:]),
        'Avg_CD': np.mean(cd_rates[-tail:]),
        'Avg_DD': np.mean(dd_rates[-tail:])
    }

    print(f" Finished {run_name} | Time: {time.time() - start_time:.1f}s | Coop: {steady_state['Avg_C_Rate']:.3f}")
    return steady_state

if __name__ == "__main__":
    
    matrix_SD = np.array([
        [0, 5],
        [1, 3]
    ])
    
    base_params = {
        'N': 1800,
        'n_signals': 100,
        'nu_s': 0.001,
        'rounds': 20000,
        'cmax': 0.5,
        'd_sigma': 0.2,
        'n_mutation_signals': 10,
        'payoff_matrix': matrix_SD
    }

    beta_values_to_test = [1]
    nu_p_values_to_test = [0.001]

    raw_parameter_sets = []

    for b in beta_values_to_test:
        p = base_params.copy()
        p['beta'] = b
        p['nu_p'] = 0.001
        raw_parameter_sets.append(p)

    for np_val in nu_p_values_to_test:
        p = base_params.copy()
        p['beta'] = 1
        p['nu_p'] = np_val
        raw_parameter_sets.append(p)

    unique_parameter_sets = []
    seen = set()
    for p in raw_parameter_sets:
        key = (p['beta'], p['nu_p'])
        if key not in seen:
            seen.add(key)
            unique_parameter_sets.append(p)

    n_jobs = -1 
    total_runs = len(unique_parameter_sets)
    
    print("="*60)
    print("="*60)

    # Executes parallel processing.
    results = Parallel(n_jobs=n_jobs)(delayed(run_simulation)(params) for params in unique_parameter_sets)



    print("="*60)
    print(" All simulations have completed!")
    print("="*60)
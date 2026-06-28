import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import softmax
import os
import pandas as sd
import time

# Establishes the base output path.
base_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_well_mixed_sd")
os.makedirs(base_output_path, exist_ok=True)

# Saves a plotted figure.
def save_plot(plot_func, data, title, xlabel, ylabel, filename, labels=None, colors=None):
    plt.figure(figsize=(10, 6))
    for i, d in enumerate(data):
        plot_func(d, label=labels[i] if labels else None, color=colors[i] if colors else None)
    if labels:
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close('all')

# Runs the well-mixed baseline simulation.
def run_baseline_well_mixed():
    # Sets simulation parameters.
    N = 1800
    rounds = 20000
    beta = 1.0
    nu_s = 0.001
    save_interval = 20
    
    # Defines the Prisoner's Dilemma payoff matrix.
    payoff_matrix = np.array([
        [0.0, 5.0], 
        [1.0, 3.0]  
    ])
    
    # Creates output directories.
    images_dir = os.path.join(base_output_path, "images")
    csv_dir = os.path.join(base_output_path, "csv_exports")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    # Initializes uniform random strategies (0 or 1).
    strategies = np.random.randint(0, 2, size=N)
    
    # Initializes tracking arrays.
    time_steps = []
    cooperation_rates, defection_rates = [], []
    cc_rates, cd_rates, dd_rates = [], [], []
    coop_avg_rewards, defect_avg_rewards = [], []
    
    start_time = time.time()
    
    # Executes the generational loop.
    for gen in range(rounds):
        scores = np.zeros(N)
        cc = cd = dd = 0
        coop_players, defect_players = [], []
        
        # Pairs the population randomly.
        indices = np.random.permutation(N)
        idx1 = indices[::2]
        idx2 = indices[1::2]
        
        a1 = strategies[idx1]
        a2 = strategies[idx2]
        
        # Computes rewards.
        r1 = payoff_matrix[a1, a2]
        r2 = payoff_matrix[a2, a1]
        
        scores[idx1] += r1
        scores[idx2] += r2
        
        # Tracks generation actions.
        total_coop_actions = a1.sum() + a2.sum()
        total_defect_actions = N - total_coop_actions
        
        cc += np.sum((a1 == 1) & (a2 == 1))
        cd += np.sum((a1 != a2))
        dd += np.sum((a1 == 0) & (a2 == 0))
        
        coop_players.extend(idx1[a1 == 1])
        coop_players.extend(idx2[a2 == 1])
        defect_players.extend(idx1[a1 == 0])
        defect_players.extend(idx2[a2 == 0])
        
        # Applies natural selection.
        exp_scores = np.exp(beta * scores - np.max(beta * scores))
        probs = exp_scores / exp_scores.sum()
        parent_indices = np.random.choice(N, size=N, p=probs)
        new_strategies = strategies[parent_indices].copy()
        
        # Applies mutation.
        mutation_mask = np.random.rand(N) < nu_s
        new_strategies[mutation_mask] ^= 1
        
        strategies = new_strategies
        
        # Saves statistics periodically.
        if gen % save_interval == 0:
            time_steps.append(gen)
            
            cooperation_rates.append(total_coop_actions / N)
            defection_rates.append(total_defect_actions / N)
            
            total_pairs = N // 2
            cc_rates.append(cc / total_pairs)
            cd_rates.append(cd / total_pairs)
            dd_rates.append(dd / total_pairs)
            
            coop_avg = scores[coop_players].mean() if coop_players else 0
            defect_avg = scores[defect_players].mean() if defect_players else 0
            coop_avg_rewards.append(coop_avg)
            defect_avg_rewards.append(defect_avg)
            
        # Prints progress.
        if (gen + 1) % 10000 == 0:
            print(f"[Well-Mixed Baseline] Generation {gen + 1}/{rounds} completed.")
            
    # Exports plot images.
    save_plot(plt.plot, [cooperation_rates, defection_rates], "Cooperation and Defection Rates", "Generation", "Rate", os.path.join(images_dir, "cooperation_defection_actions.png"), labels=['Cooperation', 'Defection'], colors=['blue', 'red'])
    save_plot(plt.plot, [cc_rates, cd_rates, dd_rates], "Game States (CC, CD, DD) Over Time", "Generation", "State Density", os.path.join(images_dir, "state_densities.png"), labels=['CC', 'CD', 'DD'], colors=['green', 'orange', 'red'])
    save_plot(plt.plot, [coop_avg_rewards, defect_avg_rewards], "Average Fitness by Strategy", "Generation", "Average Fitness", os.path.join(images_dir, "avg_rewards_by_strategy.png"), labels=['Cooperators', 'Defectors'], colors=['blue', 'red'])

    # Exports CSV data.
    sd.DataFrame({"t": time_steps, "Cooperation_Rate": cooperation_rates, "Defection_Rate": defection_rates}).to_csv(os.path.join(csv_dir, "cooperation_defection_actions.csv"), index=False)
    sd.DataFrame({"t": time_steps, "CC_Rate": cc_rates, "CD_Rate": cd_rates, "DD_Rate": dd_rates}).to_csv(os.path.join(csv_dir, "state_densities.csv"), index=False)
    sd.DataFrame({"t": time_steps, "Cooperators_Reward": coop_avg_rewards, "Defectors_Reward": defect_avg_rewards}).to_csv(os.path.join(csv_dir, "avg_rewards_by_strategy.csv"), index=False)
    
    print(f"Well-Mixed Baseline completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_baseline_well_mixed()
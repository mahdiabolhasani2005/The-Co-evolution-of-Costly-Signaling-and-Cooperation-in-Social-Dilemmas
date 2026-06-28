import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import softmax
import os
import pandas as pd
from matplotlib.animation import PillowWriter
import time

# --- Base path for saving outputs ---
base_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "single_run_output")
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
# | Helper function for vectorized sampling from categorical distributions         |
# | (equivalent to vectorized np.random.choice)                                    |
# ----------------------------------------------------------------------------------
def _sample_categorical(probs):
    """Vectorized sampling from categorical distributions along the last axis.
    Exact equivalent (distribution-wise) of np.random.choice using inverse CDF method."""
    cdf = np.cumsum(probs, axis=-1)
    cdf = cdf / cdf[..., -1:]                      # Normalization to compensate for floating-point errors
    u = np.random.rand(*probs.shape[:-1], 1)
    return (cdf < u).sum(axis=-1)

# ----------------------------------------------------------------------------------
# | Main simulation function (single run)                                          |
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
    
    payoff_matrix = params['payoff_matrix']
    run_name = f"SingleRun_CMAX_{cmax:.2f}"

    # --- Create output directories ---
    run_dir = os.path.join(base_output_path, run_name)
    output_dir = os.path.join(run_dir, "plots_and_animations")
    csv_dir = os.path.join(run_dir, "csv_data")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    print(f"Starting single run: {run_name}...")
    start_time = time.time()

    # --- Initialization ---
    signal_costs = np.random.uniform(0, cmax, size=n_signals)
    signal_costs = np.sort(signal_costs) 
    
    signal_probs = np.random.dirichlet(np.ones(n_signals), size=(L, L))
    signal_response = np.random.randint(0, 2, size=(L, L, n_signals))

    # --- Fixed index grids and neighbor offsets (precomputed for speed) ---
    R, C = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    Rp1 = (R + 1) % L          # Bottom neighbor (r+1, c)
    Cp1 = (C + 1) % L          # Right neighbor (r, c+1)
    # Offsets for 5 natural selection options: [self, up, down, left, right]
    sel_dr = np.array([0, -1, 1, 0, 0])
    sel_dc = np.array([0, 0, 0, -1, 1])

    # --- Lists for storing data ---
    time_steps = []
    cooperation_rates, defection_rates = [], []
    cooperation_rates_signals, defection_rates_signals = [], []
    cc_rates, cd_rates, dd_rates = [], [] , []
    coop_avg_rewards, defect_avg_rewards = [], []
    
    signal_usage_over_time = []
    coop_strategy_over_time = []
    signal_rewards_over_time = []
    signal_power_over_time = []
    avg_signal_cost_over_time = []
    
    signal_cost_to_reward_numer = np.zeros(n_signals)
    signal_cost_to_reward_denom = np.zeros(n_signals)
    
    signal_benefit_minus_cost_over_time = [] 
    animation_frames = []

    # --- Main Loop ---
    for gen in range(rounds):
        # =========================================================================
        # | Game phase (vectorized): Each edge once, with same logic as main loop |
        # =========================================================================
        # Independent signal sampling for all four roles of each cell (matching separate loop sampling)
        S_focal_down  = _sample_categorical(signal_probs)   # s1 on bottom edge (focal cell)
        S_emit_down   = _sample_categorical(signal_probs)   # s2 from bottom neighbor
        S_focal_right = _sample_categorical(signal_probs)   # s1 on right edge (focal cell)
        S_emit_right  = _sample_categorical(signal_probs)   # s2 from right neighbor

        # --- Bottom edge: (r,c) with (r+1,c) ---
        s1_d = S_focal_down
        s2_d = S_emit_down[Rp1, C]
        a1_d = signal_response[R, C, s2_d]          # Focal response to neighbor's signal
        a2_d = signal_response[Rp1, C, s1_d]        # Neighbor's response to focal signal
        r1_d = payoff_matrix[a1_d, a2_d]
        r2_d = payoff_matrix[a2_d, a1_d]
        f1_d = r1_d - signal_costs[s1_d]
        f2_d = r2_d - signal_costs[s2_d]

        # --- Right edge: (r,c) with (r,c+1) ---
        s1_r = S_focal_right
        s2_r = S_emit_right[R, Cp1]
        a1_r = signal_response[R, C, s2_r]
        a2_r = signal_response[R, Cp1, s1_r]
        r1_r = payoff_matrix[a1_r, a2_r]
        r2_r = payoff_matrix[a2_r, a1_r]
        f1_r = r1_r - signal_costs[s1_r]
        f2_r = r2_r - signal_costs[s2_r]

        # --- Scores: Each cell sums 4 games (2 focal + 2 neighbors) ---
        scores = f1_d + f1_r
        scores += np.roll(f2_d, 1, axis=0)          # Bottom neighbor gets f2
        scores += np.roll(f2_r, 1, axis=1)          # Right neighbor gets f2

        # --- Aggregate signal event stats (4 events per cell) ---
        signals_all = np.concatenate([s1_d.ravel(), s2_d.ravel(), s1_r.ravel(), s2_r.ravel()])
        rewards_all = np.concatenate([r1_d.ravel(), r2_d.ravel(), r1_r.ravel(), r2_r.ravel()]).astype(float)
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

        gen_total_cost = signal_costs[signals_all].sum()

        # =========================================================================
        # | Natural selection (vectorized)                                        |
        # =========================================================================
        # Stack of neighbor scores: [self, up, down, left, right] then divided by 4
        neigh_scores = np.stack([
            scores,
            np.roll(scores, 1, axis=0),
            np.roll(scores, -1, axis=0),
            np.roll(scores, 1, axis=1),
            np.roll(scores, -1, axis=1),
        ], axis=2) / 4.0

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
            time_steps.append(gen)
            
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
            
            # Record average signal cost in this generation (total cost paid divided by 4 times the number of cells)
            avg_signal_cost = gen_total_cost / (L * L * 4)
            avg_signal_cost_over_time.append(avg_signal_cost)
            
            signal_power_over_time.append(np.sum(su ** 2))
            animation_frames.append(np.mean(signal_response, axis=2))

        # Print progress to console every 100 rounds
        if gen > 0 and gen % 100 == 0:
            print(f"  Round {gen} out of {rounds} completed.")

    # --- Generate plots and GIFs (all frames, no limit) ---
    print("Generating and saving plots and GIF...")
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
    plt.close('all')

    # Heatmaps
    plot_heatmap(np.array(signal_usage_over_time).T, "Signal Usage Over Time", "Generation (x10)", "Signal Index (Ordered by Cost)", 'viridis', os.path.join(output_dir, "signal_usage_heatmap.png"), "Usage Probability")
    plot_heatmap(np.array(coop_strategy_over_time).T, "Cooperation Strategy Over Time Per Signal", "Generation (x10)", "Signal Index (Ordered by Cost)", 'YlGnBu', os.path.join(output_dir, "coop_strategy_heatmap.png"), "C Probability")
    plot_heatmap(np.array(signal_benefit_minus_cost_over_time).T, "Signal Fitness (Benefit - Cost) Over Time", "Generation (x10)", "Signal Index (Ordered by Cost)", 'magma', os.path.join(output_dir, "signal_fitness_heatmap.png"), "Benefit - Cost")

    # Line Plots and Scatters
    final_cost_to_reward_ratio = np.divide(signal_cost_to_reward_numer, signal_cost_to_reward_denom + 1e-9)
    avg_usage_per_signal = np.mean(signal_usage_over_time, axis=0)
    avg_fitness_per_signal = np.mean(signal_benefit_minus_cost_over_time, axis=0) 
    
    save_plot(plt.scatter, {'x': final_cost_to_reward_ratio, 'y': avg_usage_per_signal, 'c': avg_usage_per_signal, 'cmap': 'plasma', 'edgecolor': 'k'}, "Signal Usage vs Relative Cost to Reward", "Signal Cost / Reward Ratio", "Average Signal Usage", os.path.join(output_dir, "signal_cost_to_reward_vs_usage.png"), colorbar_label="Usage Intensity")
    save_plot(plt.scatter, {'x': avg_fitness_per_signal, 'y': avg_usage_per_signal, 'color': 'darkgreen', 'alpha': 0.7, 'edgecolor': 'black'}, "Signal Usage vs True Fitness (Benefit - Cost)", "Average Signal Fitness (Benefit - Cost)", "Average Signal Usage Density", os.path.join(output_dir, "signal_usage_vs_fitness.png"))
    save_plot(plt.scatter, {'x': signal_costs, 'y': avg_usage_per_signal, 'color': 'teal', 'alpha': 0.7, 'edgecolor': 'black'}, "Signal Usage vs Signal Cost", "Signal Cost", "Average Signal Usage", os.path.join(output_dir, "signal_usage_vs_cost.png"))
    
    save_plot(plt.plot, [cooperation_rates, defection_rates], "Cooperation and Defection Rates", "Generation", "Rate", os.path.join(output_dir, "cooperation_defection_actions.png"), labels=['Cooperation Rate', 'Defection Rate'], colors=['blue', 'red'])
    save_plot(plt.plot, [cc_rates, cd_rates, dd_rates], "Game States (CC, CD, DD) Over Time", "Generation", "State Density", os.path.join(output_dir, "state_densities.png"), labels=['CC', 'CD', 'DD'], colors=['green', 'orange', 'red'])
    save_plot(plt.plot, [coop_avg_rewards, defect_avg_rewards], "Average Fitness by Strategy", "Generation", "Average Fitness (Benefit-Cost)", os.path.join(output_dir, "avg_fitness_by_strategy.png"), labels=['Cooperators', 'Defectors'], colors=['blue', 'red'])
    save_plot(plt.plot, [avg_signal_cost_over_time], "Average Signal Cost Paid Over Time", "Generation", "Average Signal Cost", os.path.join(output_dir, "avg_signal_cost_over_time.png"), labels=['Avg Cost Paid'], colors=['purple'])

    # --- Save CSVs ---
    print("Saving CSV data files...")
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
    
    # CSV files containing column t
    pd.DataFrame({"t": time_steps, "cooperation_rates": cooperation_rates, "defection_rates": defection_rates}).to_csv(os.path.join(csv_dir, "cooperation_defection_actions.csv"), index=False)
    pd.DataFrame({"t": time_steps, "cc_rates": cc_rates, "cd_rates": cd_rates, "dd_rates": dd_rates}).to_csv(os.path.join(csv_dir, "state_densities.csv"), index=False)
    pd.DataFrame({"t": time_steps, "coop_avg_fitness": coop_avg_rewards, "defect_avg_fitness": defect_avg_rewards}).to_csv(os.path.join(csv_dir, "avg_fitness_by_strategy.csv"), index=False)
    pd.DataFrame({"t": time_steps, "avg_signal_cost": avg_signal_cost_over_time}).to_csv(os.path.join(csv_dir, "avg_signal_cost_over_time.csv"), index=False)

    # --- Save animation data to CSV ---
    # Each row = one animation frame (a sampled generation)
    # First column t = generation number, and then L*L columns = smoothed grid values of that frame
    print("Saving animation data to CSV...")
    frames_arr = np.array(animation_frames)                 # Shape: (number of frames, L, L)
    flat_frames = frames_arr.reshape(frames_arr.shape[0], L * L)
    anim_df = pd.DataFrame(flat_frames, columns=[f"px_{i}" for i in range(L * L)])
    anim_df.insert(0, "t", time_steps)
    anim_df.to_csv(os.path.join(csv_dir, "animation_frames.csv"), index=False)

    # --- Extract steady-state data ---
    tail_len = max(1, len(cooperation_rates) // 5) 
    steady_state = {
        'CMAX': cmax,
        'C': np.mean(cooperation_rates[-tail_len:]),
        'D': np.mean(defection_rates[-tail_len:]),
        'CC': np.mean(cc_rates[-tail_len:]),
        'CD': np.mean(cd_rates[-tail_len:]),
        'DD': np.mean(dd_rates[-tail_len:])
    }

    end_time = time.time()
    print(f"Run finished. Total time: {end_time - start_time:.2f} seconds.")
    return steady_state


# ----------------------------------------------------------------------------------
# | Program entry point (single run only)                                          |
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    
    matrix_SH = np.array([
        [ 1, 3], 
        [ 0, 5]  
    ])
    
    # Single run parameters
    params = {
        'L': 40, 
        'n_signals': 100, 
        'nu_p': 0.001, 
        'nu_s': 0.001, 
        'beta': 1.0, 
        'cmax': 0.5, # Just a fixed value for max cost
        'd_sigma': 0.2,
        'n_mutation_signals': 10,
        'rounds': 20000, 
        'payoff_matrix': matrix_SH
    }

    # Direct function call (no loop and no parallelization)
    result = run_simulation(params)

    print("\n" + "="*50)
    print("Simulation finished.")
    print("Average stats of final generations (steady-state):")
    for key, value in result.items():
        print(f"  {key}: {value:.4f}")
    print("="*50)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from scipy.special import softmax
import os
import pandas as pd
import time

# Establishes the base output path.
base_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_lattice_sh")
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

# Runs the lattice baseline simulation.
def run_baseline_lattice():
    # Sets simulation parameters.
    L = 40
    N = L * L
    rounds = 20000
    beta = 1.0
    nu_s = 0.001
    save_interval = 20
    
    # Defines the Prisoner's Dilemma payoff matrix.
    payoff_matrix = np.array([
        [1.0, 3.0], 
        [0.0, 5.0]  
    ])
    
    # Creates output directories.
    images_dir = os.path.join(base_output_path, "images")
    csv_dir = os.path.join(base_output_path, "csv_exports")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    # Initializes uniform random strategies on the grid.
    strategies = np.random.randint(0, 2, size=(L, L))

    # Precomputes index grids and neighbor offsets (for vectorization).
    R, C = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    sel_dr = np.array([0, -1, 1, 0, 0])   # [self, up, down, left, right]
    sel_dc = np.array([0, 0, 0, -1, 1])

    # Vectorized categorical sampler (inverse-CDF; distribution-equivalent to np.random.choice).
    def sample_categorical(probs):
        cdf = np.cumsum(probs, axis=-1)
        cdf = cdf / cdf[..., -1:]
        u = np.random.rand(*probs.shape[:-1], 1)
        return (cdf < u).sum(axis=-1)
    
    # Initializes tracking arrays.
    time_steps = []
    cooperation_rates, defection_rates = [], []
    cc_rates, cd_rates, dd_rates = [], [], []
    coop_avg_rewards, defect_avg_rewards = [], []
    animation_frames = []
    
    start_time = time.time()
    
    # Executes the generational loop.
    for gen in range(rounds):
        # =====================================================================
        # | Interaction step (vectorized): each edge once.                    |
        # =====================================================================
        # Down edge: (r,c) vs (r+1,c)
        a1_d = strategies
        a2_d = np.roll(strategies, -1, axis=0)
        r1_d = payoff_matrix[a1_d, a2_d]
        r2_d = payoff_matrix[a2_d, a1_d]
        # Right edge: (r,c) vs (r,c+1)
        a1_r = strategies
        a2_r = np.roll(strategies, -1, axis=1)
        r1_r = payoff_matrix[a1_r, a2_r]
        r2_r = payoff_matrix[a2_r, a1_r]

        # Each cell's total score from its 4 games (2 focal + 2 as neighbor).
        scores = r1_d + r1_r + np.roll(r2_d, 1, axis=0) + np.roll(r2_r, 1, axis=1)

        # Aggregate action/state statistics.
        actions_all = np.concatenate([a1_d.ravel(), a2_d.ravel(), a1_r.ravel(), a2_r.ravel()])
        rewards_all = np.concatenate([r1_d.ravel(), r2_d.ravel(), r1_r.ravel(), r2_r.ravel()])

        cc = int(((a1_d == 1) & (a2_d == 1)).sum() + ((a1_r == 1) & (a2_r == 1)).sum())
        dd = int(((a1_d == 0) & (a2_d == 0)).sum() + ((a1_r == 0) & (a2_r == 0)).sum())
        cd = 2 * L * L - cc - dd

        total_coop_actions = int((actions_all == 1).sum())
        total_defect_actions = int((actions_all == 0).sum())

        coop_mask = actions_all == 1
        coop_score_sum = rewards_all[coop_mask].sum()
        coop_count = int(coop_mask.sum())
        defect_score_sum = rewards_all[~coop_mask].sum()
        defect_count = int((~coop_mask).sum())

        # =====================================================================
        # | Local selection (vectorized).                                     |
        # =====================================================================
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

        idx_selected = sample_categorical(sel_probs)          # (L, L) in 0..4
        src_r = (R + sel_dr[idx_selected]) % L
        src_c = (C + sel_dc[idx_selected]) % L
        new_strategies = strategies[src_r, src_c]

        # Applies mutation.
        mutation_mask_s = np.random.rand(L, L) < nu_s
        new_strategies[mutation_mask_s] ^= 1
        
        strategies = new_strategies
        
        # Saves statistics periodically.
        if gen % save_interval == 0:
            time_steps.append(gen)
            
            total_interactions = L * L * 2
            cooperation_rates.append(total_coop_actions / (2 * total_interactions))
            defection_rates.append(total_defect_actions / (2 * total_interactions))
            
            cc_rates.append(cc / total_interactions)
            cd_rates.append(cd / total_interactions)
            dd_rates.append(dd / total_interactions)
            
            coop_avg_rewards.append(coop_score_sum / coop_count if coop_count else 0)
            defect_avg_rewards.append(defect_score_sum / defect_count if defect_count else 0)

            # Records a snapshot of the strategy grid for the animation (1 = C, 0 = D).
            animation_frames.append(strategies.astype(float).copy())
            
        # Prints progress.
        if (gen + 1) % 100 == 0:
            print(f"[Lattice Baseline] Generation {gen + 1}/{rounds} completed.")
            
    # =====================================================================
    # | Builds and saves the animation (cooperation map over time).        |
    # =====================================================================
    print("Building animation...")
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(animation_frames[0], cmap='viridis', vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Strategy (1=C, 0=D)")
    title = ax.set_title(f"Generation {time_steps[0]}")

    def update(frame):
        im.set_array(animation_frames[frame])
        title.set_text(f"Generation {time_steps[frame]}")
        return [im, title]

    ani = animation.FuncAnimation(fig, update, frames=len(animation_frames), interval=200, blit=True)
    gif_path = os.path.join(images_dir, "cooperation_map_animation.gif")
    ani.save(gif_path, writer=PillowWriter(fps=5))
    plt.close('all')

    # Exports plot images.
    save_plot(plt.plot, [cooperation_rates, defection_rates], "Cooperation and Defection Rates", "Generation", "Rate", os.path.join(images_dir, "cooperation_defection_actions.png"), labels=['Cooperation', 'Defection'], colors=['blue', 'red'])
    save_plot(plt.plot, [cc_rates, cd_rates, dd_rates], "Game States (CC, CD, DD) Over Time", "Generation", "State Density", os.path.join(images_dir, "state_densities.png"), labels=['CC', 'CD', 'DD'], colors=['green', 'orange', 'red'])
    save_plot(plt.plot, [coop_avg_rewards, defect_avg_rewards], "Average Fitness by Strategy", "Generation", "Average Fitness", os.path.join(images_dir, "avg_rewards_by_strategy.png"), labels=['Cooperators', 'Defectors'], colors=['blue', 'red'])

    # Exports CSV data.
    pd.DataFrame({"t": time_steps, "Cooperation_Rate": cooperation_rates, "Defection_Rate": defection_rates}).to_csv(os.path.join(csv_dir, "cooperation_defection_actions.csv"), index=False)
    pd.DataFrame({"t": time_steps, "CC_Rate": cc_rates, "CD_Rate": cd_rates, "DD_Rate": dd_rates}).to_csv(os.path.join(csv_dir, "state_densities.csv"), index=False)
    pd.DataFrame({"t": time_steps, "Cooperators_Reward": coop_avg_rewards, "Defectors_Reward": defect_avg_rewards}).to_csv(os.path.join(csv_dir, "avg_rewards_by_strategy.csv"), index=False)

    # Exports the animation data to CSV (each row = one frame; column t = generation, then L*L flattened grid values).
    frames_arr = np.array(animation_frames)                       # shape: (n_frames, L, L)
    flat_frames = frames_arr.reshape(frames_arr.shape[0], L * L)
    anim_df = pd.DataFrame(flat_frames, columns=[f"px_{i}" for i in range(L * L)])
    anim_df.insert(0, "t", time_steps)
    anim_df.to_csv(os.path.join(csv_dir, "animation_frames.csv"), index=False)

    print(f"Lattice Baseline completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_baseline_lattice()
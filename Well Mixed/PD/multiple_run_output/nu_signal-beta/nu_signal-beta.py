import numpy as np
import matplotlib
matplotlib.use('Agg') # بک‌اند غیرتعاملی برای اجرای امن موازی
import matplotlib.pyplot as plt
from scipy.special import softmax
import os
import pandas as pd
from joblib import Parallel, delayed
import time

# --- مسیر پایه برای ذخیره تمام خروجی‌های اجرای دسته‌ای ---
base_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "well_mixed_parameter_sweep")
os.makedirs(base_output_path, exist_ok=True)

# ----------------------------------------------------------------------------------
# | توابع کمکی برای رسم نمودار                                                     |
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
# | تابع اصلی شبیه‌سازی برای یک ترکیب پارامتر                                       |
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
    payoff_matrix = params['payoff_matrix']
    sample_interval = params.get('sample_interval', 50) # نمونه‌برداری هر 50 راند
    
    run_name = f"Beta_{beta}_nuP_{nu_p}"

    # ایجاد پوشه‌های مربوط به این اجرای خاص
    run_dir = os.path.join(base_output_path, run_name)
    images_dir = os.path.join(run_dir, "images")
    csv_dir = os.path.join(run_dir, "csv_exports")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    start_time = time.time()

    # --- جمعیت اولیه ---
    signal_costs = np.random.uniform(0, cmax, size=n_signals)
    signal_probs = np.random.dirichlet(np.ones(n_signals), size=N)
    signal_response = np.random.randint(0, 2, size=(N, n_signals))

    # --- آمارها ---
    cooperation_rates_signals, defection_rates_signals = [], []
    cooperation_rates, defection_rates = [], []
    cc_rates, cd_rates, dd_rates = [], [], []
    coop_avg_rewards, defect_avg_rewards = [], []
    
    signal_usage_over_time = []
    coop_strategy_over_time = []
    signal_fitness_over_time = [] 
    signal_power_over_time = []
    
    signal_cost_to_reward_numer = np.zeros(n_signals)
    signal_cost_to_reward_denom = np.zeros(n_signals)

    # --- حلقه‌ی تکاملی ---
    for gen in range(rounds):
        
        # پرینت وضعیت فقط هر 10 هزار راند برای جلوگیری از شلوغی کنسول
        if (gen + 1) % 1000 == 0:  
            print(f"[{run_name}] پیشرفت: راند {gen + 1} از {rounds} تکمیل شد.")
            
        scores = np.zeros(N)

        # جفت کردن تصادفی
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

        # فیتنس واقعی (Benefit - Cost)
        fitness1 = r1 - cost1
        fitness2 = r2 - cost2

        scores[idx1] += fitness1
        scores[idx2] += fitness2

        # =====================================================================
        # 🌟 استخراج داده‌های آماری فقط در زمان نمونه‌برداری (هر 50 راند) 🌟
        # =====================================================================
        if gen % sample_interval == 0:
            signal_usage = np.zeros(n_signals)
            signal_coop_usage = np.zeros(n_signals)
            signal_total_fitness = np.zeros(n_signals) 
            signal_counts = np.zeros(n_signals)

            # ثبت آمار سیگنال‌ها برای بازیکنان اول
            for s, gross_r, fit_val, a_other in zip(s1, r1, fitness1, a2):
                signal_usage[s] += 1
                signal_total_fitness[s] += fit_val
                signal_counts[s] += 1
                signal_coop_usage[s] += a_other
                if gross_r > 0:
                    signal_cost_to_reward_numer[s] += signal_costs[s]
                    signal_cost_to_reward_denom[s] += gross_r

            # ثبت آمار سیگنال‌ها برای بازیکنان دوم
            for s, gross_r, fit_val, a_other in zip(s2, r2, fitness2, a1):
                signal_usage[s] += 1
                signal_total_fitness[s] += fit_val
                signal_counts[s] += 1
                signal_coop_usage[s] += a_other
                if gross_r > 0:
                    signal_cost_to_reward_numer[s] += signal_costs[s]
                    signal_cost_to_reward_denom[s] += gross_r

            # آمار کل راند
            total_coop_actions = a1.sum() + a2.sum()
            total_defect_actions = 2 * len(a1) - total_coop_actions

            cc = np.sum((a1 == 1) & (a2 == 1))
            cd = np.sum((a1 != a2))
            dd = np.sum((a1 == 0) & (a2 == 0))

            coop_players = np.concatenate([idx1[a1 == 1], idx2[a2 == 1]])
            defect_players = np.concatenate([idx1[a1 == 0], idx2[a2 == 0]])

            # ذخیره آمار در لیست‌ها
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

            coop_avg = scores[coop_players].mean() if len(coop_players) > 0 else 0
            defect_avg = scores[defect_players].mean() if len(defect_players) > 0 else 0
            coop_avg_rewards.append(coop_avg)
            defect_avg_rewards.append(defect_avg)

            signal_usage_over_time.append(signal_usage / (signal_usage.sum() + 1e-9))
            coop_strategy_over_time.append(safe_divide(signal_coop_usage, signal_counts))
            signal_fitness_over_time.append(safe_divide(signal_total_fitness, signal_counts))

            norm_usage = signal_usage / N
            signal_power = np.sum(norm_usage ** 2)
            signal_power_over_time.append(signal_power)
        # =====================================================================

        # --- انتخاب طبیعی (این بخش در همه راندها اجرا می‌شود) ---
        exp_scores = np.exp(beta * scores - np.max(beta * scores))
        probs = exp_scores / exp_scores.sum()
        parent_indices = np.random.choice(N, size=N, p=probs)
        new_signal_probs = signal_probs[parent_indices].copy()
        new_signal_response = signal_response[parent_indices].copy()

        # --- جهش (این بخش در همه راندها اجرا می‌شود) ---
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

    # -------------------------------------------------------------
    # تولید نمودارها
    # -------------------------------------------------------------
    final_cost_to_reward_ratio = safe_divide(signal_cost_to_reward_numer, signal_cost_to_reward_denom)
    avg_usage_per_signal = np.mean(signal_usage_over_time, axis=0)
    avg_fitness_per_signal = np.mean(signal_fitness_over_time, axis=0)

    # آپدیت لیبل‌ها به x50
    plot_heatmap(np.array(signal_usage_over_time).T, "Signal Usage Over Time", f"Generation (x{sample_interval})", "Signal Index", 'viridis', os.path.join(images_dir, "signal_usage_heatmap.png"), "Usage Probability")
    plot_heatmap(np.array(coop_strategy_over_time).T, "Cooperation Strategy Over Time Per Signal", f"Generation (x{sample_interval})", "Signal Index", 'YlGnBu', os.path.join(images_dir, "coop_strategy_heatmap.png"), "C Probability")
    plot_heatmap(np.array(signal_fitness_over_time).T, "Signal Fitness (Benefit - Cost) Over Time", f"Generation (x{sample_interval})", "Signal Index", 'magma', os.path.join(images_dir, "signal_fitness_heatmap.png"), "Benefit - Cost")

    save_plot(plt.scatter, {'x': final_cost_to_reward_ratio, 'y': avg_usage_per_signal, 'c': avg_usage_per_signal, 'cmap': 'plasma', 'edgecolor': 'k'}, "Signal Usage vs Relative Cost to Reward", "Signal Cost / Gross Reward Ratio", "Average Signal Usage", os.path.join(images_dir, "signal_cost_to_reward_vs_usage.png"), colorbar_label="Usage Intensity")
    save_plot(plt.scatter, {'x': avg_fitness_per_signal, 'y': avg_usage_per_signal, 'color': 'darkgreen', 'alpha': 0.7, 'edgecolor': 'black'}, "Signal Usage vs True Fitness (Benefit - Cost)", "Average Signal Fitness (Benefit - Cost)", "Average Signal Usage Density", os.path.join(images_dir, "signal_usage_vs_fitness.png"))
    save_plot(plt.scatter, {'x': signal_costs, 'y': avg_usage_per_signal, 'color': 'teal', 'alpha': 0.7, 'edgecolor': 'black'}, "Signal Usage vs Signal Cost", "Signal Cost", "Average Signal Usage", os.path.join(images_dir, "signal_usage_vs_cost.png"))

    save_plot(plt.plot, [cooperation_rates_signals, defection_rates_signals], "Cooperation and Defection Rates in Strategies", f"Generation (x{sample_interval})", "Rate", os.path.join(images_dir, "cooperation_defection_strategies.png"), labels=['Cooperation', 'Defection'], colors=['blue', 'red'])
    save_plot(plt.plot, [cooperation_rates, defection_rates], "Cooperation and Defection Rates (Actions)", f"Generation (x{sample_interval})", "Rate", os.path.join(images_dir, "cooperation_defection_actions.png"), labels=['Cooperation', 'Defection'], colors=['blue', 'red'])
    save_plot(plt.plot, [cc_rates, cd_rates, dd_rates], "Game States (CC, CD, DD) Over Time", f"Generation (x{sample_interval})", "State Density", os.path.join(images_dir, "state_densities.png"), labels=['CC', 'CD', 'DD'], colors=['green', 'orange', 'red'])
    save_plot(plt.plot, [coop_avg_rewards, defect_avg_rewards], "Average Fitness by Strategy", f"Generation (x{sample_interval})", "Average Fitness", os.path.join(images_dir, "avg_rewards_by_strategy.png"), labels=['Cooperators', 'Defectors'], colors=['blue', 'red'])
    save_plot(plt.plot, [signal_power_over_time], "Signal Usage Concentration Over Time", f"Generation (x{sample_interval})", "Sum of Squared Signal Usage", os.path.join(images_dir, "signal_power_over_time.png"), colors=['purple'])

    # -------------------------------------------------------------
    # تولید CSVها
    # -------------------------------------------------------------
    pd.DataFrame({"Cooperation_Rate_Strategies": cooperation_rates_signals, "Defection_Rate_Strategies": defection_rates_signals}).to_csv(os.path.join(csv_dir, "coop_defect_strategies.csv"), index=False)
    pd.DataFrame({"Cooperation_Rate": cooperation_rates, "Defection_Rate": defection_rates}).to_csv(os.path.join(csv_dir, "cooperation_defection_actions.csv"), index=False)
    pd.DataFrame({"CC_Rate": cc_rates, "CD_Rate": cd_rates, "DD_Rate": dd_rates}).to_csv(os.path.join(csv_dir, "state_densities.csv"), index=False)
    pd.DataFrame({"Cooperators_Reward": coop_avg_rewards, "Defectors_Reward": defect_avg_rewards}).to_csv(os.path.join(csv_dir, "avg_rewards_by_strategy.csv"), index=False)
    pd.DataFrame({"Signal_Power": signal_power_over_time}).to_csv(os.path.join(csv_dir, "signal_power.csv"), index=False)
    
    pd.DataFrame(signal_usage_over_time, columns=[f"Sig_{i}" for i in range(n_signals)]).to_csv(os.path.join(csv_dir, "signal_usage_over_time.csv"), index=False)
    pd.DataFrame(coop_strategy_over_time, columns=[f"Sig_{i}" for i in range(n_signals)]).to_csv(os.path.join(csv_dir, "coop_strategy_over_time.csv"), index=False)
    pd.DataFrame(signal_fitness_over_time, columns=[f"Sig_{i}" for i in range(n_signals)]).to_csv(os.path.join(csv_dir, "signal_fitness_over_time.csv"), index=False)

    pd.DataFrame({
        "signal_cost": signal_costs,
        "avg_fitness_benefit_minus_cost": avg_fitness_per_signal, 
        "avg_usage_per_signal": avg_usage_per_signal,
        "final_cost_to_reward_ratio": final_cost_to_reward_ratio
    }).to_csv(os.path.join(csv_dir, "signal_metrics_summary.csv"), index=False)

    # میانگین 20 درصد آخر (از داده‌های نمونه‌برداری شده) برای حالت پایدار
    tail = max(1, len(cooperation_rates) // 5)
    steady_state = {
        'Beta': beta,
        'nu_p': nu_p,
        'Avg_C_Rate': np.mean(cooperation_rates[-tail:]),
        'Avg_D_Rate': np.mean(defection_rates[-tail:]),
        'Avg_CC': np.mean(cc_rates[-tail:]),
        'Avg_CD': np.mean(cd_rates[-tail:]),
        'Avg_DD': np.mean(dd_rates[-tail:])
    }

    print(f"✅ پایان {run_name} | زمان: {time.time() - start_time:.1f}s | Coop: {steady_state['Avg_C_Rate']:.3f}")
    return steady_state


if __name__ == "__main__":
    
    matrix_SD = np.array([
        [1, 5],
        [0, 3]
    ])
    
    base_params = {
        'N': 1800,
        'n_signals': 100,
        'nu_s': 0.001,
        'rounds': 100000,
        'cmax': 0.5,
        'd_sigma': 0.2,
        'n_mutation_signals': 10,
        'sample_interval': 50, # 🌟 داده‌ها هر 50 راند ثبت می‌شوند 🌟
        'payoff_matrix': matrix_SD
    }

    # مقادیری که درخواست کرده بودید
    beta_values_to_test = [0.01, 0.1, 1, 10, 50, 100]
    nu_p_values_to_test = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

    raw_parameter_sets = []

    # 🌟 بخش اول: nu_p ثابت = 0.001، بتا 6 مقدار مختلف 🌟
    for b in beta_values_to_test:
        p = base_params.copy()
        p['beta'] = b
        p['nu_p'] = 0.001
        raw_parameter_sets.append(p)

    # 🌟 بخش دوم: بتا ثابت = 1، nu_p شش مقدار مختلف 🌟
    for np_val in nu_p_values_to_test:
        p = base_params.copy()
        p['beta'] = 1
        p['nu_p'] = np_val
        raw_parameter_sets.append(p)

    # فیلتر تکراری‌ها (مجموعاً 11 شبیه‌سازی منحصر‌به‌فرد می‌شود)
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
    print(f"شروع اجرای موازی {total_runs} شبیه‌سازی منحصر‌به‌فرد ...")
    print(f"نمونه‌برداری: هر {base_params['sample_interval']} راند یکبار")
    print("="*60)

    # اجرای موازی
    results = Parallel(n_jobs=n_jobs)(delayed(run_simulation)(params) for params in unique_parameter_sets)

    # ذخیره فایل خلاصه نهایی
    master_df = pd.DataFrame(results)
    master_df = master_df.sort_values(by=['Beta', 'nu_p']).reset_index(drop=True)
    master_csv_path = os.path.join(base_output_path, "master_sweep_summary.csv")
    master_df.to_csv(master_csv_path, index=False)

    print("="*60)
    print("🎉 تمام شبیه‌سازی‌ها به پایان رسیدند!")
    print(f"فایل خلاصه نهایی در مسیر زیر ذخیره شد:\n{master_csv_path}")
    print("="*60)
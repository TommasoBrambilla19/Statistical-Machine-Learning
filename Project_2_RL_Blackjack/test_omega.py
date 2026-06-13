import blackjack_extended as bjk
import blackjack_base as bjk_base
from math import inf
import RL as rl
import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

def read_reward_file(filepath):
    """
    Legge il file csv generato (episode,avg_reward) e restituisce due liste:
    episodes, rewards
    """
    episodes = []
    rewards = []
    if not os.path.exists(filepath):
        return None, None
        
    with open(filepath, 'r') as f:
        # Salta l'header
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                episodes.append(int(parts[0]))
                rewards.append(float(parts[1]))
    return episodes, rewards

def plot_omega_comparison(directory, plot_dir, method_name, file_prefix, decks, omega_values):
    """
    Crea un grafico sovrapposto per i diversi valori di omega.
    """
    plt.figure(figsize=(10, 6))
    
    for omega in omega_values:
        filename = "{}_{}_omega_{}.txt".format(file_prefix, decks, omega)
        filepath = os.path.join(directory, filename)
        
        episodes, rewards = read_reward_file(filepath)
        
        if episodes and rewards:
            plt.plot(episodes, rewards, label=f"$\omega={omega}$")
    
    plt.title(f"Confronto Omega - {method_name} ({decks} Decks)")
    plt.xlabel("Episodi")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)
    
    # Salva il grafico
    output_filename = f"tuning_{method_name.replace(' ', '_')}_decks_{decks}.png"
    output_path = os.path.join(plot_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] Salvato grafico: {output_filename}")

if __name__ == "__main__":
    # Setup directories
    directory = "{}/data".format(sys.path[0])
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plot_dir = "{}/figures/".format(sys.path[0])
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # --- CONSTANTS ---
    n_sims = 10 ** 4       # Increase to 10**5 or 10**6 for better convergence!
    epsilon = 0.05
    init_val = 0.0
    warmup = n_sims // 10
    
    # PARAMETER TUNING
    # Omega controls the learning rate decay: alpha = 1 / n^omega
    # Theoretical range for convergence: 0.5 < omega <= 1.0
    omega_values = [0.5, 0.65, 0.77, 0.85, 1.0]
    
    # Deck configurations to test
    deck_configs = [1, 2, 6, 8, inf]

    # Storage for results
    # Structure: { deck_count: { omega: {'q_ext': val, 'q_sum': val, 'mc': val} } }
    results = {}

    print(f"Starting simulation with {n_sims} episodes per run...")

    for decks in deck_configs:
        results[decks] = {}
        print(f"\n===== DECKS: {decks} =====")
        
        # Set seed & Init Envs
        seed = 31233
        env = bjk.BlackjackEnvExtend(decks=decks, seed=seed)
        sum_env = bjk_base.BlackjackEnvBase(decks=decks, seed=seed)

        # 1. RUN MONTE CARLO (Baseline - Omega does not affect MC)
        print(f"  [MC] Running Baseline...")
        # Path for MC file (standardized)
        mc_path = "{}/{}_{}.txt".format(directory, "hand_MC_state", decks)
        
        start_mc = time.time()
        Q_MC, MC_avg_reward, _ = rl.learn_MC(
            env, n_sims, gamma=1, epsilon=epsilon, init_val=init_val,
            episode_file=mc_path, warmup=warmup
        )
        print(f"  [MC] Finished. Avg Reward: {MC_avg_reward:.5f}")
        
        # 2. LOOP OVER OMEGA VALUES
        for omega in omega_values:
            print(f"  [Q-Learning] Testing Omega = {omega}")
            results[decks][omega] = {'mc': MC_avg_reward} # Store baseline for comparison
            
            # --- Q-Learning Extended ---
            # Unique filename for this omega
            file_path_ext = "{}/{}_{}_omega_{}.txt".format(directory, "hand_state", decks, omega)
            
            Q_ext, q_ext_reward, _ = rl.learn_Q(
                env, n_sims, gamma=1, omega=omega, epsilon=epsilon, init_val=init_val,
                episode_file=file_path_ext, warmup=warmup
            )
            results[decks][omega]['q_ext'] = q_ext_reward
            
            # --- Q-Learning Sum ---
            file_path_sum = "{}/{}_{}_omega_{}.txt".format(directory, "sum_state", decks, omega)
            
            Q_sum, q_sum_reward, _ = rl.learn_Q(
                sum_env, n_sims, omega=omega, epsilon=epsilon, init_val=init_val,
                episode_file=file_path_sum, warmup=warmup
            )
            results[decks][omega]['q_sum'] = q_sum_reward

        # --- GENERATE PLOTS FOR THIS DECK CONFIGURATION ---
        print(f"  [Plotting] Generazione grafici di confronto per {decks} decks...")
        
        # Plot 1: Q-Extended (confronto omega)
        plot_omega_comparison(
            directory=directory, 
            plot_dir=plot_dir, 
            method_name="Q_Extended", 
            file_prefix="hand_state", 
            decks=decks, 
            omega_values=omega_values
        )
        
        # Plot 2: Q-Sum (confronto omega)
        plot_omega_comparison(
            directory=directory, 
            plot_dir=plot_dir, 
            method_name="Q_Sum", 
            file_prefix="sum_state", 
            decks=decks, 
            omega_values=omega_values
        )

    # --- GENERATE OUTPUT TABLES ---

    print("\n\n" + "="*90)
    print(f"{'RESULTS SUMMARY (Avg Reward)':^90}")
    print("="*90)
    
    # Print a table for each Deck configuration
    for decks in deck_configs:
        print(f"\nDecks: {decks}")
        print("-" * 65)
        print(f"{'Omega':<10} | {'Q-Extended':<15} | {'Q-Sum':<15} | {'MC (Ref)':<15}")
        print("-" * 65)
        for omega in omega_values:
            row = results[decks][omega]
            print(f"{omega:<10} | {row['q_ext']:<15.5f} | {row['q_sum']:<15.5f} | {row['mc']:<15.5f}")
        print("-" * 65)

    # --- GENERATE LATEX TABLE ---
    # We will create a consolidated Latex table
    tex_filename = "omega_tuning_results.tex"
    with open(tex_filename, "w") as f:
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\begin{document}\n\n")
        
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Impact of learning rate decay parameter $\\omega$ on Average Reward (" + str(n_sims) + " sims)}\n")
        f.write("\\label{tab:omega_tuning}\n")
        f.write("\\begin{tabular}{l|ccc|ccc}\n")
        f.write("\\toprule\n")
        f.write(" & \\multicolumn{3}{c|}{\\textbf{Q-Extended Space}} & \\multicolumn{3}{c}{\\textbf{Q-Sum Space}} \\\\\n")
        f.write("\\textbf{Omega} & 1 Deck & 6 Decks & Inf & 1 Deck & 6 Decks & Inf \\\\\n")
        f.write("\\midrule\n")
        
        # We assume 1, 6, and Inf are present in deck_configs for this table construction
        # If they aren't, this part might need adjustment, but your list includes them.
        target_decks = [1, 6, inf]
        
        for omega in omega_values:
            line = f"{omega}"
            # Add Extended results
            for d in target_decks:
                if d in results:
                    val = results[d][omega]['q_ext']
                    line += f" & {val:.4f}"
                else:
                    line += " & -"
            
            # Add Sum results
            for d in target_decks:
                if d in results:
                    val = results[d][omega]['q_sum']
                    line += f" & {val:.4f}"
                else:
                    line += " & -"
            
            line += " \\\\\n"
            f.write(line)
            
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
        f.write("\\end{document}\n")

    print(f"\n[INFO] LaTeX table generated: {tex_filename}")
    print("="*90)
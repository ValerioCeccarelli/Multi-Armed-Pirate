"""
Esempio di utilizzo della funzione plot_conversion_rates

Questo script dimostra come utilizzare il nuovo grafico di conversion rate
implementato in plotting.py
"""

import numpy as np
from matplotlib import pyplot as plt

from environments import StochasticEnvironment
from agents import CombinatorialUCBBidding
from baselines import FixedActionBaselineAgent
from plotting import plot_conversion_rates

# Configurazione simulazione
def main():
    print("=== Esempio Conversion Rate Plot ===")
    
    # Parametri
    num_trials = 5
    time_horizon = 500
    num_items = 2
    budget = 200
    prices = np.linspace(0.1, 1.0, 10)
    
    print(f"Configurazione:")
    print(f"- Numero trials: {num_trials}")
    print(f"- Time horizon: {time_horizon}")
    print(f"- Numero items: {num_items}")
    print(f"- Budget: {budget}")
    print(f"- Prezzi: {prices}")
    
    # Crea ambiente
    def create_env():
        return StochasticEnvironment(
            distribution_functions=[
                StochasticEnvironment.gaussian_distribution(mean=0.4, std=0.15),
                StochasticEnvironment.gaussian_distribution(mean=0.6, std=0.15),
            ],
            num_rounds=time_horizon
        )
    
    # Simula
    print(f"\nEsecuzione simulazioni...")
    
    # Raccogli dati da multiple simulazioni
    all_valuations = []
    all_agent_arms = []
    all_baseline_arms = []
    
    for trial in range(num_trials):
        print(f"  Trial {trial + 1}/{num_trials}")
        
        env = create_env()
        agent = CombinatorialUCBBidding(
            num_items=num_items,
            price_set=prices,
            budget=budget,
            time_horizon=time_horizon,
            alpha=1.0
        )
        baseline = FixedActionBaselineAgent(
            num_items=num_items,
            prices=prices,
            time_horizon=time_horizon,
            valuations=env.valuations,
            budget=budget
        )
        
        # Simula agent
        agent_arms = np.full((num_items, time_horizon), -1, dtype=np.int64)
        baseline_arms = np.full((num_items, time_horizon), -1, dtype=np.int64)
        
        for t in range(time_horizon):
            # Agent
            agent_choice = agent.pull_arm()
            if np.all(agent_choice == -1):
                break
                
            # Baseline
            baseline_choice = baseline.pull_arm()
            
            # Environment
            valuations_t = env.round(t)
            
            # Calcola rewards
            agent_rewards = np.zeros(num_items)
            baseline_rewards = np.zeros(num_items)
            
            for i in range(num_items):
                if agent_choice[i] >= 0 and valuations_t[i] >= prices[agent_choice[i]]:
                    agent_rewards[i] = prices[agent_choice[i]]
                if baseline_choice[i] >= 0 and valuations_t[i] >= prices[baseline_choice[i]]:
                    baseline_rewards[i] = prices[baseline_choice[i]]
            
            # Update agents
            agent.update(agent_rewards)
            baseline.update(baseline_rewards)
            
            # Store
            agent_arms[:, t] = agent_choice
            baseline_arms[:, t] = baseline_choice
        
        all_valuations.append(env.valuations.T)  # (time_horizon, num_items)
        all_agent_arms.append(agent_arms.T)  # (time_horizon, num_items)
        all_baseline_arms.append(baseline_arms.T)  # (time_horizon, num_items)
    
    # Organizza dati per plotting
    valuations = np.stack(all_valuations, axis=0)  # (num_trials, time_horizon, num_items)
    valuations = valuations.transpose(0, 2, 1)  # (num_trials, num_items, time_horizon)
    
    agent_played_arms = np.stack(all_agent_arms, axis=0)  # (num_trials, time_horizon, num_items)
    agent_played_arms = agent_played_arms.transpose(0, 2, 1)  # (num_trials, num_items, time_horizon)
    agent_played_arms = agent_played_arms[np.newaxis, ...]  # (1, num_trials, num_items, time_horizon)
    
    baseline_played_arms = np.stack(all_baseline_arms, axis=0)  # (num_trials, time_horizon, num_items)
    baseline_played_arms = baseline_played_arms.transpose(0, 2, 1)  # (num_trials, num_items, time_horizon)
    
    print(f"\nCreazione grafico conversion rate...")
    print(f"- Forme dati: valuations={valuations.shape}, agent_arms={agent_played_arms.shape}, baseline_arms={baseline_played_arms.shape}")
    
    # Crea il plot (ora con dual subplots automatici)
    plot_conversion_rates(
        valuations=valuations,
        agents_played_arms=agent_played_arms,
        baseline_played_arms=baseline_played_arms,
        prices=prices,
        agents_names=["Combinatorial UCB"],
        save_plot=True,
        save_path="example_conversion_rates.png"
    )
    
    print(f"\nConversion rate plot creato!")
    print(f"Il grafico mostra DUE subplot separati:")
    print(f"- SUBPLOT SINISTRO (Baseline): Conversion rates del baseline ottimale")
    print(f"- SUBPLOT DESTRO (Agente): Conversion rates dell'agente che impara")
    print(f"- Linee continue vs tratteggiate: items diversi")
    print(f"- Colori diversi: items diversi") 
    print(f"- Asse Y: percentuale di successo (0-100%)")
    print(f"- Asse X: prezzo")
    print(f"\nQuesto permette un confronto diretto tra strategia ottimale e appresa!")
    print(f"Il file 'example_conversion_rates.png' è stato salvato.")

if __name__ == "__main__":
    main()

"""
Test per confrontare gli agenti in ambiente non stazionario.

Questo script confronta l'agente standard con quello sliding window
in un ambiente che cambia nel tempo, per dimostrare l'efficacia
dell'approccio sliding window in condizioni non stazionarie.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time

from src.agent import CombinatorialUCBBidding, SlidingWindowUCBBidding
from src.core import ExperimentConfig, EnvironmentConfig
from src.environment import AbruptSlightlyNonstationaryEnvironment


def setup_reproducible_environment(seed: int = 42) -> None:
    """Configura ambiente per risultati riproducibili."""
    random.seed(seed)
    np.random.seed(seed)


def run_agent_in_nonstationary_env(agent_class, env: AbruptSlightlyNonstationaryEnvironment, 
                                  price_range: List[float], budget: int, 
                                  exploration_param: float = 0.3,
                                  window_size: int = None) -> Dict:
    """
    Esegue una simulazione con un agente in ambiente non stazionario.
    
    Returns:
        Dict con risultati dettagliati della simulazione
    """
    # Crea agente
    if agent_class == SlidingWindowUCBBidding:
        agent = agent_class(
            num_items=env.num_items,
            price_set=price_range,
            budget=budget,
            time_horizon=env.time_horizon,
            window_size=window_size,
            exploration_param=exploration_param
        )
        agent_name = f"SlidingWindow(T={window_size})"
    else:
        agent = agent_class(
            num_items=env.num_items,
            price_set=price_range,
            budget=budget,
            time_horizon=env.time_horizon,
            exploration_param=exploration_param
        )
        agent_name = "Standard"
    
    # Traccia risultati
    results = {
        'agent_name': agent_name,
        'total_rewards': [],
        'cumulative_rewards': [],
        'price_choices': [],
        'interval_performance': {},  # Performance per interval
        'rounds_played': 0,
        'final_budget': budget
    }
    
    cumulative_reward = 0
    
    # Esegue simulazione round per round
    for round_num in range(env.time_horizon):
        if agent.remaining_budget <= 0:
            break
            
        # Seleziona prezzi
        chosen_price_indices = agent.select_prices()
        chosen_prices = [price_range[idx] for idx in chosen_price_indices]
        
        # Ottieni decisioni di acquisto dall'ambiente
        purchase_decisions = env.get_purchase_decisions(round_num, chosen_prices)
        
        # Calcola rewards e costs
        rewards = np.array([
            price if purchased else 0 
            for price, purchased in zip(chosen_prices, purchase_decisions)
        ])
        costs = np.array(purchase_decisions, dtype=int)
        
        # Aggiorna agente
        agent.update_statistics(chosen_price_indices, rewards, costs)
        
        # Traccia risultati
        round_reward = np.sum(rewards)
        cumulative_reward += round_reward
        
        results['total_rewards'].append(round_reward)
        results['cumulative_rewards'].append(cumulative_reward)
        results['price_choices'].append(chosen_price_indices.copy())
        
        # Traccia performance per intervallo
        current_interval = env.get_current_interval_info(round_num)
        interval_key = f"interval_{current_interval.start_round}_{current_interval.end_round}"
        
        if interval_key not in results['interval_performance']:
            results['interval_performance'][interval_key] = {
                'total_reward': 0,
                'rounds': 0,
                'start_round': current_interval.start_round,
                'end_round': current_interval.end_round,
                'means': current_interval.means.copy(),
                'stds': current_interval.stds.copy()
            }
        
        results['interval_performance'][interval_key]['total_reward'] += round_reward
        results['interval_performance'][interval_key]['rounds'] += 1
    
    results['rounds_played'] = len(results['total_rewards'])
    results['final_budget'] = agent.remaining_budget
    results['total_cumulative_reward'] = cumulative_reward
    
    return results


def compare_agents_nonstationary(env: AbruptSlightlyNonstationaryEnvironment, 
                                price_range: List[float], 
                                budget: int,
                                window_sizes: List[int] = [50, 100, 150]) -> Dict:
    """
    Confronta diversi agenti nell'ambiente non stazionario.
    """
    print("CONFRONTO AGENTI IN AMBIENTE NON STAZIONARIO")
    print("=" * 60)
    print(env.get_interval_summary())
    
    all_results = {}
    
    # Test agente standard
    print("Testing agente standard...")
    setup_reproducible_environment(seed=456)
    standard_results = run_agent_in_nonstationary_env(
        CombinatorialUCBBidding, env, price_range, budget
    )
    all_results['standard'] = standard_results
    
    # Test agenti sliding window con diverse dimensioni
    for window_size in window_sizes:
        print(f"Testing agente sliding window (T={window_size})...")
        setup_reproducible_environment(seed=456)  # Stesso seed per fair comparison
        
        sliding_results = run_agent_in_nonstationary_env(
            SlidingWindowUCBBidding, env, price_range, budget, window_size=window_size
        )
        all_results[f'sliding_{window_size}'] = sliding_results
    
    return all_results


def analyze_results(results: Dict, env: AbruptSlightlyNonstationaryEnvironment) -> None:
    """Analizza e stampa i risultati del confronto."""
    print("\nRISULTATI FINALI:")
    print("=" * 60)
    
    # Riassunto generale
    for agent_key, result in results.items():
        print(f"\n{result['agent_name']}:")
        print(f"  - Reward totale: {result['total_cumulative_reward']:.2f}")
        print(f"  - Rounds giocati: {result['rounds_played']}")
        print(f"  - Budget finale: {result['final_budget']:.2f}")
        print(f"  - Reward medio per round: {result['total_cumulative_reward']/result['rounds_played']:.3f}")
    
    # Trova il migliore
    best_agent = max(results.keys(), key=lambda k: results[k]['total_cumulative_reward'])
    best_reward = results[best_agent]['total_cumulative_reward']
    
    print(f"\n🏆 MIGLIOR AGENTE: {results[best_agent]['agent_name']}")
    print(f"   Reward totale: {best_reward:.2f}")
    
    # Confronto per intervalli
    print(f"\nPERFORMANCE PER INTERVALLO:")
    print("-" * 40)
    
    # Ottieni tutti gli intervalli
    sample_result = list(results.values())[0]
    intervals = sorted(sample_result['interval_performance'].keys())
    
    for interval_key in intervals:
        interval_info = sample_result['interval_performance'][interval_key]
        print(f"\nIntervallo {interval_info['start_round']}-{interval_info['end_round']-1}:")
        print(f"  Means: {[f'{m:.2f}' for m in interval_info['means']]}")
        
        for agent_key, result in results.items():
            if interval_key in result['interval_performance']:
                interval_perf = result['interval_performance'][interval_key]
                avg_reward = interval_perf['total_reward'] / interval_perf['rounds']
                print(f"  {result['agent_name']:20s}: {avg_reward:.3f} reward/round")


def plot_performance_comparison(results: Dict, env: AbruptSlightlyNonstationaryEnvironment) -> None:
    """Crea grafici per confrontare le performance."""
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Reward cumulativo nel tempo
    plt.subplot(2, 2, 1)
    for agent_key, result in results.items():
        plt.plot(result['cumulative_rewards'], label=result['agent_name'], linewidth=2)
    
    # Aggiungi linee verticali per gli intervalli
    for interval in env.intervals[1:]:  # Skip first interval
        plt.axvline(x=interval.start_round, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Round')
    plt.ylabel('Reward Cumulativo')
    plt.title('Reward Cumulativo nel Tempo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Reward per round (moving average)
    plt.subplot(2, 2, 2)
    window = 50
    for agent_key, result in results.items():
        rewards = result['total_rewards']
        if len(rewards) >= window:
            moving_avg = [np.mean(rewards[i:i+window]) for i in range(len(rewards)-window+1)]
            plt.plot(range(window-1, len(rewards)), moving_avg, 
                    label=result['agent_name'], linewidth=2)
    
    # Aggiungi linee verticali per gli intervalli
    for interval in env.intervals[1:]:
        plt.axvline(x=interval.start_round, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Round')
    plt.ylabel(f'Reward Medio (finestra {window})')
    plt.title('Reward per Round (Media Mobile)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Performance per intervallo
    plt.subplot(2, 2, 3)
    sample_result = list(results.values())[0]
    intervals = sorted(sample_result['interval_performance'].keys())
    
    agent_names = [result['agent_name'] for result in results.values()]
    interval_labels = [f"Int {i+1}" for i in range(len(intervals))]
    
    x = np.arange(len(intervals))
    width = 0.15
    
    for i, (agent_key, result) in enumerate(results.items()):
        interval_rewards = []
        for interval_key in intervals:
            if interval_key in result['interval_performance']:
                interval_perf = result['interval_performance'][interval_key]
                avg_reward = interval_perf['total_reward'] / interval_perf['rounds']
                interval_rewards.append(avg_reward)
            else:
                interval_rewards.append(0)
        
        plt.bar(x + i*width, interval_rewards, width, 
               label=result['agent_name'], alpha=0.8)
    
    plt.xlabel('Intervallo')
    plt.ylabel('Reward Medio per Round')
    plt.title('Performance per Intervallo')
    plt.xticks(x + width * (len(results)-1)/2, interval_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Distribuzione dei reward
    plt.subplot(2, 2, 4)
    for agent_key, result in results.items():
        rewards = result['total_rewards']
        # Rimuovi zeri per una migliore visualizzazione
        non_zero_rewards = [r for r in rewards if r > 0]
        if non_zero_rewards:
            plt.hist(non_zero_rewards, bins=20, alpha=0.6, 
                    label=result['agent_name'], density=True)
    
    plt.xlabel('Reward per Round')
    plt.ylabel('Densità')
    plt.title('Distribuzione Reward (esclusi zeri)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nonstationary_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Funzione principale per eseguire il test completo."""
    print("TEST AGENTI IN AMBIENTE NON STAZIONARIO")
    print("=" * 60)
    
    # Crea ambiente non stazionario drammatico
    l = []
    for _ in range(10):
        l.extend([10, 20])

    s = []
    for _ in range(10):
        s.extend([1, 1])

    env = AbruptSlightlyNonstationaryEnvironment(
        mean=[l, l, l, l],
        std=[s, s, s, s],
        time_horizon=800
    )
    
    # Parametri esperimento
    price_range = list(range(5, 31, 5))  # [5, 10, 15, 20, 25, 30]
    budget = 400
    window_sizes = [10, 20, 30, 50, 100, 200]  # Diverse dimensioni sliding window
    
    print(f"Parametri esperimento:")
    print(f"- Prezzi disponibili: {price_range}")
    print(f"- Budget totale: {budget}")
    print(f"- Window sizes testati: {window_sizes}")
    print(f"- Time horizon: {env.time_horizon}")
    
    # Esegui confronto
    start_time = time.time()
    results = compare_agents_nonstationary(env, price_range, budget, window_sizes)
    end_time = time.time()
    
    print(f"\nTempo di esecuzione: {end_time - start_time:.2f} secondi")
    
    # Analizza risultati
    analyze_results(results, env)
    
    # Crea grafici
    print("\nGenerando grafici...")
    plot_performance_comparison(results, env)
    
    print("\nTest completato! Grafici salvati come 'nonstationary_comparison.png'")


if __name__ == "__main__":
    main()

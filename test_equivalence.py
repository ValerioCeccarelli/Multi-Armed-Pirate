"""
Test per verificare l'equivalenza tra CombinatorialUCBBidding e SlidingWindowUCBBidding
quando la sliding window è infinita (uguale al time horizon).
"""

import random
import numpy as np
from typing import List, Tuple

from src.agent import CombinatorialUCBBidding, SlidingWindowUCBBidding
from src.core import ExperimentConfig, EnvironmentConfig, run_single_simulation
from src.environment import Environment


def setup_reproducible_environment(seed: int = 42) -> None:
    """Configura ambiente per risultati riproducibili."""
    random.seed(seed)
    np.random.seed(seed)


def create_test_config() -> ExperimentConfig:
    """Crea configurazione di test semplificata."""
    env_config = EnvironmentConfig(
        means=[15, 20],  # 2 items
        stds=[5, 6],
    )
    
    experiment_config = ExperimentConfig(
        time_horizon=100,  # Horizon ridotto per test veloce
        budget=50,
        price_range=list(range(5, 26, 5)),  # [5, 10, 15, 20, 25]
        num_trials=1,
        exploration_param=0.3,
        environment=env_config
    )
    
    return experiment_config


def run_agent_simulation(agent_class, config: ExperimentConfig, 
                        window_size: int = None) -> Tuple[List, List, List]:
    """
    Esegue una simulazione con un agente specifico.
    
    Args:
        agent_class: Classe dell'agente da testare
        config: Configurazione dell'esperimento
        window_size: Dimensione della sliding window (solo per SlidingWindowUCBBidding)
        
    Returns:
        Tuple di (price_choices, rewards, costs) per ogni round
    """
    # Crea ambiente
    env = Environment(
        mean=config.environment.means,
        std=config.environment.stds,
        time_horizon=config.time_horizon
    )
    
    # Crea agente
    if agent_class == SlidingWindowUCBBidding:
        agent = agent_class(
            num_items=config.num_items,
            price_set=config.price_range,
            budget=config.budget,
            time_horizon=config.time_horizon,
            window_size=window_size,
            exploration_param=config.exploration_param
        )
    else:
        agent = agent_class(
            num_items=config.num_items,
            price_set=config.price_range,
            budget=config.budget,
            time_horizon=config.time_horizon,
            exploration_param=config.exploration_param
        )
    
    price_choices = []
    rewards_history = []
    costs_history = []
    
    # Esegue simulazione round per round
    for round_num in range(config.time_horizon):
        if agent.remaining_budget <= 0:
            break
            
        # Seleziona prezzi
        chosen_price_indices = agent.select_prices()
        chosen_prices = [config.price_range[idx] for idx in chosen_price_indices]
        
        # Ottieni decisioni di acquisto dall'ambiente
        purchase_decisions = env.get_purchase_decisions(round_num, chosen_prices)
        
        # Calcola rewards e costs come nel sistema originale
        rewards = np.array([
            price if purchased else 0 
            for price, purchased in zip(chosen_prices, purchase_decisions)
        ])
        costs = np.array(purchase_decisions, dtype=int)  # 1 per acquisto, 0 altrimenti
        
        # Aggiorna agente
        agent.update_statistics(chosen_price_indices, rewards, costs)
        
        # Salva dati
        price_choices.append(chosen_price_indices.copy())
        rewards_history.append(rewards.copy())
        costs_history.append(costs.copy())
    
    return price_choices, rewards_history, costs_history


def compare_agents_detailed(config: ExperimentConfig) -> None:
    """
    Confronta dettagliatamente i due agenti con window size infinita.
    """
    print("CONFRONTO DETTAGLIATO TRA GLI AGENTI")
    print("=" * 50)
    
    # Imposta seed per riproducibilità
    setup_reproducible_environment(seed=123)
    
    # Esegui simulazione con agente originale
    print("Eseguendo simulazione con CombinatorialUCBBidding...")
    original_prices, original_rewards, original_costs = run_agent_simulation(
        CombinatorialUCBBidding, config
    )
    
    # Reset seed per stessa sequenza casuale
    setup_reproducible_environment(seed=123)
    
    # Esegui simulazione con sliding window agente (window = time_horizon)
    print("Eseguendo simulazione con SlidingWindowUCBBidding (window infinita)...")
    sliding_prices, sliding_rewards, sliding_costs = run_agent_simulation(
        SlidingWindowUCBBidding, config, window_size=config.time_horizon
    )
    
    # Confronta risultati
    print(f"\nNumero di round eseguiti:")
    print(f"- Agente originale: {len(original_prices)}")
    print(f"- Agente sliding window: {len(sliding_prices)}")
    
    # Verifica se le scelte di prezzo sono identiche
    prices_identical = True
    min_rounds = min(len(original_prices), len(sliding_prices))
    
    for round_num in range(min_rounds):
        if not np.array_equal(original_prices[round_num], sliding_prices[round_num]):
            prices_identical = False
            print(f"\nDifferenza al round {round_num}:")
            print(f"- Originale: {original_prices[round_num]}")
            print(f"- Sliding: {sliding_prices[round_num]}")
            break
    
    if prices_identical and len(original_prices) == len(sliding_prices):
        print("\n✅ SUCCESSO: Le scelte di prezzo sono IDENTICHE!")
    else:
        print("\n❌ ERRORE: Le scelte di prezzo sono DIVERSE!")
    
    # Confronta rewards e costs totali
    total_original_rewards = sum(np.sum(r) for r in original_rewards)
    total_sliding_rewards = sum(np.sum(r) for r in sliding_rewards)
    total_original_costs = sum(np.sum(c) for c in original_costs)
    total_sliding_costs = sum(np.sum(c) for c in sliding_costs)
    
    print(f"\nReward totali:")
    print(f"- Agente originale: {total_original_rewards:.4f}")
    print(f"- Agente sliding window: {total_sliding_rewards:.4f}")
    print(f"- Differenza: {abs(total_original_rewards - total_sliding_rewards):.6f}")
    
    print(f"\nCosti totali:")
    print(f"- Agente originale: {total_original_costs:.4f}")
    print(f"- Agente sliding window: {total_sliding_costs:.4f}")
    print(f"- Differenza: {abs(total_original_costs - total_sliding_costs):.6f}")
    
    # Mostra primi 10 round per debug
    print(f"\nPrimi 10 round di scelte (indici prezzi):")
    print("Round | Originale | Sliding")
    print("-" * 30)
    for i in range(min(10, min_rounds)):
        orig_str = str(original_prices[i])
        slid_str = str(sliding_prices[i])
        print(f"{i:5d} | {orig_str:9s} | {slid_str}")


def test_different_window_sizes(config: ExperimentConfig) -> None:
    """
    Testa l'agente sliding window con diverse dimensioni di finestra.
    """
    print("\n\nTEST CON DIVERSE DIMENSIONI DI FINESTRA")
    print("=" * 50)
    
    window_sizes = [10, 25, 50, config.time_horizon]  # Ultima è infinita
    
    for window_size in window_sizes:
        setup_reproducible_environment(seed=456)
        
        print(f"\nTesting window size: {window_size}")
        prices, rewards, costs = run_agent_simulation(
            SlidingWindowUCBBidding, config, window_size=window_size
        )
        
        total_rewards = sum(np.sum(r) for r in rewards)
        total_costs = sum(np.sum(c) for c in costs)
        rounds_played = len(prices)
        
        print(f"- Rounds giocati: {rounds_played}")
        print(f"- Reward totali: {total_rewards:.4f}")
        print(f"- Costi totali: {total_costs:.4f}")
        print(f"- Efficienza (reward/costo): {total_rewards/total_costs:.4f}")


def main():
    """Funzione principale per eseguire tutti i test."""
    print("TESTING EQUIVALENZA SLIDING WINDOW vs AGENTE ORIGINALE")
    print("=" * 60)
    
    config = create_test_config()
    
    print("Configurazione del test:")
    print(f"- Time horizon: {config.time_horizon}")
    print(f"- Budget: {config.budget}")
    print(f"- Numero items: {config.num_items}")
    print(f"- Prezzi disponibili: {config.price_range}")
    print(f"- Exploration param: {config.exploration_param}")
    print(f"- Means: {config.environment.means}")
    print(f"- Stds: {config.environment.stds}")
    
    # Test principale: confronto con window infinita
    compare_agents_detailed(config)
    
    # Test aggiuntivo: diverse dimensioni di finestra
    test_different_window_sizes(config)
    
    print("\n" + "=" * 60)
    print("Test completati!")


if __name__ == "__main__":
    main()

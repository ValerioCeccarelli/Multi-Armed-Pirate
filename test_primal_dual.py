#!/usr/bin/env python3
"""
Test semplice per verificare che PrimalDualAgent funzioni correttamente.
"""

import numpy as np
from agents import PrimalDualAgent, UCBAgent

def test_primal_dual_agent():
    """Test base per PrimalDualAgent"""
    print("=== Test PrimalDualAgent ===")
    
    # Parametri del test
    num_prices = 5
    budget = 50.0
    horizon = 100
    
    # Crea l'agente
    agent = PrimalDualAgent(num_prices=num_prices, budget=budget, horizon=horizon)
    
    print(f"Agente creato con {num_prices} prezzi, budget {budget}, orizzonte {horizon}")
    print(f"Rho (pacing rate): {agent.rho}")
    print(f"Eta (learning rate): {agent.eta}")
    print(f"Lambda iniziale: {agent.lmbd}")
    
    # Simula alcuni round
    total_reward = 0.0
    total_cost = 0.0
    
    for round in range(20):
        # Pull arm
        action = agent.pull_arm()
        price_index = action[0]
        
        if price_index == -1:
            print(f"Round {round}: Budget esaurito")
            break
        
        # Simula reward (casuale per il test)
        # Simula che prezzi più bassi abbiano più probabilità di vendita
        sale_prob = max(0.1, 1.0 - (price_index / num_prices))
        sale = np.random.random() < sale_prob
        reward = (price_index + 1) * 0.2 if sale else 0.0  # Prezzo simulato
        
        # Update agent
        agent.update(np.array([reward]))
        
        total_reward += reward
        total_cost += (1.0 if reward > 0 else 0.0)
        
        print(f"Round {round}: prezzo_idx={price_index}, vendita={sale}, reward={reward:.3f}, "
              f"lambda={agent.lmbd:.3f}, budget_rimanente={agent.remaining_budget:.1f}")
    
    print(f"\nRisultati finali:")
    print(f"Reward totale: {total_reward:.3f}")
    print(f"Unità vendute: {total_cost}")
    print(f"Budget rimanente: {agent.remaining_budget}")
    print(f"Lambda finale: {agent.lmbd:.3f}")

def test_comparison_with_ucb():
    """Confronto rapido con UCBAgent"""
    print("\n=== Confronto con UCBAgent ===")
    
    num_prices = 4
    
    # UCB Agent
    ucb_agent = UCBAgent(num_prices=num_prices)
    
    # PrimalDual Agent  
    pd_agent = PrimalDualAgent(num_prices=num_prices, budget=30.0, horizon=50)
    
    print("Entrambi gli agenti creati correttamente!")
    
    # Test che entrambi possano fare pull_arm
    ucb_action = ucb_agent.pull_arm()
    pd_action = pd_agent.pull_arm()
    
    print(f"UCB action: {ucb_action}")
    print(f"PrimalDual action: {pd_action}")
    
    # Test update
    test_reward = np.array([0.5])
    ucb_agent.update(test_reward)
    pd_agent.update(test_reward)
    
    print("Update completati con successo per entrambi!")

if __name__ == "__main__":
    np.random.seed(42)  # Per risultati riproducibili
    test_primal_dual_agent()
    test_comparison_with_ucb()
    print("\n✅ Tutti i test completati con successo!")

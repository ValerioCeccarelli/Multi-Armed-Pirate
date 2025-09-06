#!/usr/bin/env python3
"""
Test per verificare che MultiProductPrimalDualAgent funzioni correttamente.
"""

import numpy as np
from agents import MultiProductPrimalDualAgent, UCBAgent

def test_multiproduct_primal_dual_agent():
    """Test base per MultiProductPrimalDualAgent"""
    print("=== Test MultiProductPrimalDualAgent ===")
    
    # Parametri del test
    num_items = 3
    num_prices = 4
    budget = 60.0
    horizon = 100
    
    # Crea l'agente
    agent = MultiProductPrimalDualAgent(
        num_items=num_items, 
        num_prices=num_prices, 
        budget=budget, 
        horizon=horizon
    )
    
    print(f"Agente creato con {num_items} items, {num_prices} prezzi per item")
    print(f"Budget: {budget}, Orizzonte: {horizon}")
    print(f"Rho (pacing rate per item): {agent.rho}")
    print(f"Eta (learning rate): {agent.eta}")
    print(f"Lambda iniziale: {agent.lmbd}")
    
    # Simula alcuni round
    total_revenue = 0.0
    total_sales = 0
    
    for round in range(25):
        # Pull arms
        actions = agent.pull_arm()
        
        if any(idx == -1 for idx in actions):
            print(f"Round {round}: Budget esaurito")
            break
        
        # Simula rewards per ogni item
        rewards = []
        sales = []
        
        for i, price_idx in enumerate(actions):
            # Simula che prezzi più bassi abbiano più probabilità di vendita
            # e che diversi item abbiano diverse propensioni alla vendita
            base_prob = max(0.1, 1.0 - (price_idx / num_prices))
            item_modifier = 1.0 - (i * 0.1)  # Item 0 vende di più, item 2 di meno
            sale_prob = base_prob * item_modifier
            
            sale = np.random.random() < sale_prob
            # Prezzo simulato: più alto l'indice, più alto il prezzo
            simulated_price = (price_idx + 1) * 0.25
            reward = simulated_price if sale else 0.0
            
            rewards.append(reward)
            sales.append(sale)
        
        rewards_array = np.array(rewards)
        
        # Update agent
        agent.update(rewards_array)
        
        total_revenue += np.sum(rewards_array)
        total_sales += sum(sales)
        
        print(f"Round {round}: actions={actions}, vendite={sales}, "
              f"rewards={[f'{r:.2f}' for r in rewards]}, "
              f"lambda={agent.lmbd:.3f}, budget_rimanente={agent.remaining_budget:.1f}")
    
    print(f"\nRisultati finali:")
    print(f"Revenue totale: {total_revenue:.3f}")
    print(f"Unità vendute totali: {total_sales}")
    print(f"Budget rimanente: {agent.remaining_budget}")
    print(f"Lambda finale: {agent.lmbd:.3f}")

def test_multiproduct_vs_single():
    """Confronto tra MultiProduct e singoli agenti"""
    print("\n=== Confronto MultiProduct vs Singoli Agenti ===")
    
    num_items = 2
    num_prices = 3
    
    # MultiProduct Agent
    mp_agent = MultiProductPrimalDualAgent(
        num_items=num_items, 
        num_prices=num_prices, 
        budget=20.0, 
        horizon=30
    )
    
    print(f"MultiProduct Agent creato: {num_items} items, {num_prices} prezzi")
    print(f"Rho per item: {mp_agent.rho}")
    
    # Test pull_arm
    mp_actions = mp_agent.pull_arm()
    print(f"MultiProduct actions: {mp_actions}")
    print(f"Shape: {mp_actions.shape}, dtype: {mp_actions.dtype}")
    
    # Test update
    test_rewards = np.array([0.3, 0.0])  # Solo il primo item vende
    mp_agent.update(test_rewards)
    print(f"Update completato. Lambda: {mp_agent.lmbd:.3f}")
    print(f"Budget rimanente: {mp_agent.remaining_budget}")

def test_edge_cases():
    """Test casi limite"""
    print("\n=== Test Casi Limite ===")
    
    # Test con budget esaurito
    agent = MultiProductPrimalDualAgent(
        num_items=2, 
        num_prices=3, 
        budget=1.0,  # Budget molto basso
        horizon=10
    )
    
    # Forza esaurimento budget
    agent.remaining_budget = 0.5
    
    actions = agent.pull_arm()
    print(f"Con budget basso ({agent.remaining_budget}): actions = {actions}")
    
    # Test con reward nullo
    test_rewards = np.array([0.0, 0.0])
    agent.update(test_rewards)
    print(f"Dopo reward nulli: lambda = {agent.lmbd:.3f}")
    
    # Test assertion
    try:
        bad_agent = MultiProductPrimalDualAgent(
            num_items=0,  # Dovrebbe fallire
            num_prices=3,
            budget=10.0,
            horizon=20
        )
    except AssertionError as e:
        print(f"Assertion correttamente catturata: {e}")
    
    try:
        bad_agent = MultiProductPrimalDualAgent(
            num_items=2,
            num_prices=1,  # Dovrebbe fallire
            budget=10.0,
            horizon=20
        )
    except AssertionError as e:
        print(f"Assertion correttamente catturata: {e}")

if __name__ == "__main__":
    np.random.seed(42)  # Per risultati riproducibili
    test_multiproduct_primal_dual_agent()
    test_multiproduct_vs_single()
    test_edge_cases()
    print("\n✅ Tutti i test MultiProduct completati con successo!")

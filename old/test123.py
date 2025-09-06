from src.agent import SlidingWindowUCBBidding
import numpy as np

# Test di base per verificare che l'agente funzioni
agent = SlidingWindowUCBBidding(
    num_items=2, 
    price_set=[1.0, 2.0, 3.0], 
    budget=100, 
    time_horizon=50, 
    window_size=10
)

print('Agente creato con successo!')
print(f'Numero items: {agent.num_items}')
print(f'Numero prezzi: {agent.num_prices}')
print(f'Window size: {agent.window_size}')

# Test della selezione iniziale
prices = agent.select_prices()
print(f'Primi prezzi selezionati: {prices}')

# Test dell\'update
rewards = np.array([0.5, 0.8])
costs = np.array([1.0, 2.0])
agent.update_statistics(prices, rewards, costs)

print('Update completato con successo!')
print(f'Round corrente: {agent.current_round}')
print(f'Budget rimanente: {agent.remaining_budget}')
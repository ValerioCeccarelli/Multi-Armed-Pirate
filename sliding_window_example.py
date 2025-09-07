"""
Esempio di utilizzo del CombinatorialUCBBiddingSlidingWindow

Questo script mostra come utilizzare la versione con sliding window 
dell'agente CombinatorialUCBBidding per ambienti non-stazionari.
"""

import numpy as np
from environments import StochasticEnvironment
from agents import CombinatorialUCBBiddingSlidingWindow

def create_nonstationary_environment(time_horizon: int, num_items: int = 2, shift_point: int = None):
    """
    Crea un ambiente non-stazionario con un cambio di distribuzione a metà simulazione.
    
    Args:
        time_horizon: Orizzonte temporale
        num_items: Numero di items nell'ambiente
        shift_point: Punto in cui avviene il cambio (default: metà dell'orizzonte)
    """
    if shift_point is None:
        shift_point = time_horizon // 2
    
    class NonstationaryEnvironment(StochasticEnvironment):
        def __init__(self):
            # Iniziamo con distribuzioni per tutti gli items
            initial_distributions = []
            for i in range(num_items):
                # Ogni item ha una distribuzione leggermente diversa
                mean_val = 0.3 + i * 0.1  # Item 0: 0.3, Item 1: 0.4, etc.
                initial_distributions.append(
                    StochasticEnvironment.gaussian_distribution(mean=mean_val, std=0.1)
                )
            
            super().__init__(
                distribution_functions=initial_distributions,
                num_rounds=time_horizon
            )
            self.shift_point = shift_point
            self.current_round = 0
            
        def round(self, t: int):
            self.current_round = t
            # Cambio di distribuzione a metà simulazione
            if t >= self.shift_point:
                # Nuove distribuzioni con mean più alte per tutti gli items
                new_distributions = []
                for i in range(self.num_items):
                    # Shift verso l'alto delle distribuzioni
                    new_mean = 0.6 + i * 0.1  # Item 0: 0.6, Item 1: 0.7, etc.
                    new_distributions.append(
                        StochasticEnvironment.gaussian_distribution(mean=new_mean, std=0.1)
                    )
                self.distribution_functions = new_distributions
            
            return super().round(t)
    
    return NonstationaryEnvironment()

def main():
    print("=== Esempio CombinatorialUCBBiddingSlidingWindow ===")
    
    # Parametri della simulazione
    num_trials = 3
    time_horizon = 1000
    num_items = 2
    budget = 500
    window_size = 100  # Sliding window di 100 osservazioni
    
    # Set di prezzi
    prices = np.linspace(0.1, 1.0, 10)
    print(f"Set di prezzi: {prices}")
    
    # Crea agente con sliding window
    agent = CombinatorialUCBBiddingSlidingWindow(
        num_items=num_items,
        price_set=prices,
        budget=budget,
        time_horizon=time_horizon,
        window_size=window_size,
        alpha=1.0
    )
    
    print(f"\nParametri simulazione:")
    print(f"- Numero items: {num_items}")
    print(f"- Budget iniziale: {budget}")
    print(f"- Dimensione sliding window: {window_size}")
    print(f"- Orizzonte temporale: {time_horizon}")
    print(f"- Cambio ambiente a round: {time_horizon // 2}")
    print(f"- Distribuzioni iniziali: Item 0 (mean=0.3), Item 1 (mean=0.4)")
    print(f"- Distribuzioni finali: Item 0 (mean=0.6), Item 1 (mean=0.7)")
    
    # Simula alcune iterazioni
    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1} ---")
        
        # Crea ambiente non-stazionario
        env = create_nonstationary_environment(time_horizon, num_items)
        
        # Reset agente per nuovo trial
        agent = CombinatorialUCBBiddingSlidingWindow(
            num_items=num_items,
            price_set=prices,
            budget=budget,
            time_horizon=time_horizon,
            window_size=window_size,
            alpha=1.0
        )
        
        total_reward = 0
        total_purchases = 0
        
        for t in range(min(200, time_horizon)):  # Simuliamo solo i primi 200 round per esempio
            # Agente sceglie prezzi
            price_indices = agent.pull_arm()
            
            if np.all(price_indices == -1):
                print(f"Budget esaurito al round {t}")
                break
                
            # Ambiente genera valutazioni
            valuations = env.round(t)
            
            # Calcola rewards
            rewards = np.zeros(num_items)
            for i in range(num_items):
                if price_indices[i] >= 0:
                    price = prices[price_indices[i]]
                    if valuations[i] >= price:
                        rewards[i] = price
                        total_purchases += 1
            
            total_reward += np.sum(rewards)
            
            # Aggiorna agente
            agent.update(rewards)
            
            # Mostra statistiche ogni 50 rounds
            if (t + 1) % 50 == 0:
                print(f"  Round {t+1}: Reward totale = {total_reward:.2f}, "
                      f"Budget rimanente = {agent.remaining_budget}, "
                      f"Acquisti totali = {total_purchases}")
        
        # Statistiche finali del trial
        print(f"Trial {trial + 1} completato:")
        print(f"  - Reward totale: {total_reward:.2f}")
        print(f"  - Budget rimanente: {agent.remaining_budget}")
        print(f"  - Acquisti totali: {total_purchases}")
        
        # Mostra alcune statistiche delle sliding windows
        if trial == 0:  # Solo per il primo trial
            print(f"\n  Esempio statistiche sliding window:")
            stats = agent.get_window_statistics()
            for i in range(min(2, num_items)):  # Mostra solo primi 2 items
                for j in range(min(3, len(prices))):  # Mostra solo primi 3 prezzi
                    if stats[(i, j)]['count'] > 0:
                        print(f"    Item {i}, Prezzo {j} (€{prices[j]:.2f}): "
                              f"Count={stats[(i, j)]['count']}, "
                              f"Mean reward={stats[(i, j)]['mean_reward']:.3f}, "
                              f"Mean cost={stats[(i, j)]['mean_cost']:.3f}")

if __name__ == "__main__":
    main()

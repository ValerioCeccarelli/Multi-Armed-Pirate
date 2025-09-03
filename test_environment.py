"""
Test semplice per verificare il funzionamento dell'ambiente non stazionario.
"""

import numpy as np
from src.nonstationary_environment import create_dramatic_nonstationary_environment, create_mild_nonstationary_environment


def test_environment_basic():
    """Test di base per verificare che l'ambiente funzioni."""
    print("TEST AMBIENTE NON STAZIONARIO")
    print("=" * 40)
    
    # Crea ambiente
    env = create_dramatic_nonstationary_environment(num_items=2, time_horizon=300)
    
    print("Ambiente creato con successo!")
    print(f"Numero items: {env.num_items}")
    print(f"Time horizon: {env.time_horizon}")
    print(f"Numero intervalli: {len(env.intervals)}")
    
    # Mostra summary
    print("\n" + env.get_interval_summary())
    
    # Test alcune decisioni di acquisto
    print("TEST DECISIONI DI ACQUISTO:")
    print("-" * 30)
    
    test_prices = [10.0, 15.0]
    test_rounds = [0, 50, 100, 150, 200, 250]
    
    for round_num in test_rounds:
        if round_num < env.time_horizon:
            decisions = env.get_purchase_decisions(round_num, test_prices)
            interval_info = env.get_current_interval_info(round_num)
            print(f"Round {round_num:3d}: prezzi {test_prices} -> acquisti {decisions}")
            print(f"           Intervallo means: {[f'{m:.2f}' for m in interval_info.means]}")
    
    print("\n✅ Test ambiente completato con successo!")


def test_environment_mild():
    """Test dell'ambiente con cambiamenti più moderati."""
    print("\n\nTEST AMBIENTE MILD NON STAZIONARIO")
    print("=" * 40)
    
    env = create_mild_nonstationary_environment(num_items=3, time_horizon=400)
    
    print("Ambiente mild creato!")
    print(env.get_interval_summary())
    
    # Verifica che le variazioni siano più contenute
    max_mean_change = 0
    max_std_change = 0
    
    for i in range(1, len(env.intervals)):
        current = env.intervals[i]
        previous = env.intervals[i-1]
        
        for j in range(env.num_items):
            mean_change = abs((current.means[j] - previous.means[j]) / previous.means[j])
            std_change = abs((current.stds[j] - previous.stds[j]) / previous.stds[j])
            
            max_mean_change = max(max_mean_change, mean_change)
            max_std_change = max(max_std_change, std_change)
    
    print(f"Massima variazione means: {max_mean_change:.1%}")
    print(f"Massima variazione stds: {max_std_change:.1%}")
    print("✅ Test ambiente mild completato!")


if __name__ == "__main__":
    test_environment_basic()
    test_environment_mild()

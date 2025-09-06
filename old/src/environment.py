import numpy as np
from typing import List, NamedTuple


class IntervalInfo(NamedTuple):
    """Informazioni su un intervallo dell'ambiente non stazionario."""
    start_round: int
    end_round: int
    means: List[float]
    stds: List[float]


class Environment:
    """
    Simulation environment for the multi-item pricing problem.
    
    Simulates user valuations for different items over time,
    determining whether a user will purchase an item given its price.
    """
    
    def __init__(self, mean: List[float], std: List[float], time_horizon: int):
        """
        Initialize the environment with valuation distributions.
        
        Args:
            mean: List of valuation means for each item
            std: List of standard deviations for each item  
            time_horizon: Number of time rounds to simulate
            
        Raises:
            ValueError: If input parameters are invalid
        """
        self._validate_inputs(mean, std, time_horizon)
        
        self.num_items = len(mean)
        self.time_horizon = time_horizon
        self.mean_valuations = np.array(mean)
        self.std_valuations = np.array(std)
        
        # Valuations matrix (items, time)
        # Each row represents valuations for a specific item over time
        self.valuations = np.array([
            np.random.normal(m, s, size=time_horizon) 
            for m, s in zip(mean, std)
        ])
    
    def _validate_inputs(self, mean: List[float], std: List[float], time_horizon: int) -> None:
        """
        Validate input parameters for the environment.
        
        Args:
            mean: List of valuation means
            std: List of standard deviations
            time_horizon: Time horizon
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not mean or not std:
            raise ValueError("Mean and std lists cannot be empty")
        
        if len(mean) != len(std):
            raise ValueError("Mean and std lists must have the same length")
        
        if any(s <= 0 for s in std):
            raise ValueError("All standard deviations must be positive")
        
        if time_horizon <= 0:
            raise ValueError("Time horizon must be positive")

    def get_purchase_decisions(self, round_idx: int, prices: List[float]) -> List[bool]:
        """
        Determine if users will purchase items at given prices.
        
        Args:
            round_idx: Current round index (0-indexed)
            prices: List of prices for each item
            
        Returns:
            List of purchase decisions (True = purchase, False = no purchase)
            
        Raises:
            IndexError: If round_idx is out of valid range
            ValueError: If number of prices doesn't match number of items
        """
        if round_idx < 0 or round_idx >= self.time_horizon:
            raise IndexError(f"Round index {round_idx} out of range [0, {self.time_horizon-1}]")
        
        if len(prices) != self.num_items:
            raise ValueError(f"Number of prices ({len(prices)}) must match number of items ({self.num_items})")
        
        # A user purchases if price <= their valuation for that item
        return [price <= valuation for price, valuation in zip(prices, self.valuations[:, round_idx])]
    
    # Compatibility method with existing interface
    def round(self, t: int, prices_t: List[float]) -> List[bool]:
        """Alias for get_purchase_decisions for compatibility."""
        return self.get_purchase_decisions(t, prices_t)


class AbruptSlightlyNonstationaryEnvironment:
    """
    Simulation environment for the multi-item pricing problem.
    
    Simulates user valuations for different items over time,
    determining whether a user will purchase an item given its price.
    """
    def __init__(self, mean: list[list[float]], std: list[list[float]], time_horizon: int):
        """
        Initialize the environment with valuation distributions.
        
        Args:
            means: List of valuation means for each item
            stds: List of standard deviations for each item  
            time_horizon: Number of time rounds to simulate
            num_intervals: Number of intervals to simulate
            
        Raises:
            ValueError: If input parameters are invalid
        """        
        self.num_items = len(mean)
        self.time_horizon = time_horizon

        num_intervals = len(mean[0])
        print([len(std[i]) == num_intervals for i in range(self.num_items)])
        assert all(len(mean[i]) == num_intervals for i in range(self.num_items)), "All means lists must have the same length"
        assert all(len(std[i]) == num_intervals for i in range(self.num_items)), "All stds lists must have the same length"
        assert len(mean) == len(std), "All means and stds lists must have the same length"
        assert len(mean[0]) == len(std[0]), "All means and stds lists must have the same length"
        assert time_horizon % num_intervals == 0, "Time horizon must be divisible by number of intervals"

        self.num_intervals = num_intervals
        self.interval_duration = time_horizon // num_intervals
        self.mean_valuations = np.array(mean)
        self.std_valuations = np.array(std)

        # Crea lista di intervalli per compatibilità con il test
        self.intervals = []
        for i in range(num_intervals):
            start_round = i * self.interval_duration
            end_round = (i + 1) * self.interval_duration
            interval_means = [mean[j][i] for j in range(self.num_items)]
            interval_stds = [std[j][i] for j in range(self.num_items)]
            self.intervals.append(IntervalInfo(start_round, end_round, interval_means, interval_stds))

        valuations = [[] for _ in range(self.num_items)]
        for i in range(num_intervals):
            for j in range(self.num_items):
                for _ in range(self.interval_duration):
                    valuations[j].append(np.random.normal(mean[j][i], std[j][i]))
        self.valuations = np.array(valuations)

    def get_purchase_decisions(self, round_idx: int, prices: List[float]) -> List[bool]:
        """
        Determine if users will purchase items at given prices.
        
        Args:
            round_idx: Current round index (0-indexed)
            prices: List of prices for each item
            
        Returns:
            List of purchase decisions (True = purchase, False = no purchase)
            
        Raises:
            IndexError: If round_idx is out of valid range
            ValueError: If number of prices doesn't match number of items
        """
        if round_idx < 0 or round_idx >= self.time_horizon:
            raise IndexError(f"Round index {round_idx} out of range [0, {self.time_horizon-1}]")
        
        if len(prices) != self.num_items:
            raise ValueError(f"Number of prices ({len(prices)}) must match number of items ({self.num_items})")
        
        # A user purchases if price <= their valuation for that item
        return [price <= valuation for price, valuation in zip(prices, self.valuations[:, round_idx])]
    
    # Compatibility method with existing interface
    def round(self, t: int, prices_t: List[float]) -> List[bool]:
        """Alias for get_purchase_decisions for compatibility."""
        return self.get_purchase_decisions(t, prices_t)
    
    def get_current_interval_info(self, round_idx: int) -> IntervalInfo:
        """
        Ottieni informazioni sull'intervallo corrente per un dato round.
        
        Args:
            round_idx: Indice del round corrente
            
        Returns:
            IntervalInfo con i dati dell'intervallo corrente
        """
        if round_idx < 0 or round_idx >= self.time_horizon:
            raise IndexError(f"Round index {round_idx} out of range [0, {self.time_horizon-1}]")
        
        interval_idx = round_idx // self.interval_duration
        return self.intervals[interval_idx]
    
    def get_interval_summary(self) -> str:
        """
        Ottieni un riassunto testuale degli intervalli dell'ambiente.
        
        Returns:
            Stringa con il riassunto degli intervalli
        """
        summary = f"Ambiente non stazionario con {self.num_intervals} intervalli:\n"
        for i, interval in enumerate(self.intervals):
            summary += f"  Intervallo {i+1} (rounds {interval.start_round}-{interval.end_round-1}):\n"
            summary += f"    Means: {[f'{m:.2f}' for m in interval.means]}\n"
            summary += f"    Stds:  {[f'{s:.2f}' for s in interval.stds]}\n"
        return summary


if __name__ == "__main__":
    means = [[3, 50, 7], [10, 150, 20]]
    stds = [[1, 2, 3], [4, 5, 6]]
    time_horizon = 30
    num_intervals = 3
    env = AbruptSlightlyNonstationaryEnvironment(means, stds, time_horizon, num_intervals)
    print(env.valuations)
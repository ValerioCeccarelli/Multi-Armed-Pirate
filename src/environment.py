import numpy as np
from typing import List


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

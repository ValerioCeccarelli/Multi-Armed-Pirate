from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np


class Agent(ABC):
    @abstractmethod
    def pull_arm(self) -> NDArray[np.int64]:
        """Select the price index (or -1 if budget depleted) for each item. (num_items,)"""
        pass

    @abstractmethod
    def update(self, rewards: NDArray[np.float64], full_rewards: NDArray[np.float64] = None) -> None:
        """Update the agent's knowledge based on the results of the round.

        Args:
            rewards: Rewards received for each item. (num_items,)
            full_rewards: Full rewards (valuations) for each item. (num_items, num_prices) Only used in some agents.
        """
        pass


class UCBAgent(Agent):
    """
    Upper Confidence Bound (UCB) agent for multi-armed bandit problems.
    This agent only supports a single item (num_items=1).
    """

    def __init__(self, num_prices: int, alpha: float = 1.0):
        assert num_prices > 1, "UCB agent requires at least 2 prices."
        self.num_prices = num_prices
        self.alpha = alpha

        # Number of times each price was selected
        self.counts = np.zeros(num_prices, dtype=np.int64)
        # Estimated value for each price
        self.values = np.zeros(num_prices, dtype=np.float64)
        self.total_counts = 0  # Total number of pulls

        self._last_price_index = -1

    def pull_arm(self) -> NDArray[np.int64]:
        if self.total_counts < self.num_prices:
            # Play each price once initially
            price_index = self.total_counts
        else:
            # Calculate UCB for each price
            ucb_values = self.values + self.alpha * np.sqrt(
                (2 * np.log(self.total_counts)) / (self.counts + 1e-5))
            price_index = int(np.argmax(ucb_values))

        self._last_price_index = price_index
        return np.array([price_index], dtype=np.int64)

    def update(self, rewards: NDArray[np.float64], full_rewards: NDArray[np.float64] = None) -> None:
        assert self._last_price_index != -1, "pull_arm must be called before update."

        reward = rewards[0]
        self.total_counts += 1
        self.counts[self._last_price_index] += 1
        self.values[self._last_price_index] += (
            reward - self.values[self._last_price_index]) / self.counts[self._last_price_index]

        self._last_price_index = -1  # Reset for next round

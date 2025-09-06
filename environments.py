

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from numpy.typing import NDArray


class Environment(ABC):
    @abstractmethod
    def round(self, round: int) -> NDArray[np.float64]:
        """Return the valuations for each item at the given round. (num_items,)"""
        pass

    @property
    @abstractmethod
    def valuations(self) -> NDArray[np.float64]:
        """Valuations matrix (num_items, time_horizon)"""
        pass

    @property
    @abstractmethod
    def time_horizon(self) -> int:
        """Total number of rounds in the environment."""
        pass

    @property
    @abstractmethod
    def num_items(self) -> int:
        """Number of items in the environment."""
        pass


class StochasticEnvironment(Environment):
    def __init__(self, distribution_func: Callable, num_items: int, num_rounds: int, seed: int = 42):
        """
        Stochastic environment where valuations are drawn from a specified distribution.

        Args:
            distribution_func (Callable): Function to generate random valuations.
            num_items (int): Number of items.
            num_rounds (int): Total number of rounds.
            seed (int): Random seed for reproducibility.
        """
        self._num_items = num_items
        self._num_rounds = num_rounds
        self._rng = np.random.default_rng(seed)
        self._valuations = self._rng.choice(
            distribution_func(), size=(num_items, num_rounds))

    @classmethod
    def gaussian_distribution(cls, mean: float = 0.5, std: float = 0.1) -> Callable[[], float]:
        """Generate a Gaussian distribution function."""
        def distribution():
            return np.random.normal(loc=mean, scale=std, size=1000)
        return distribution

    @classmethod
    def beta_distribution(cls, a: float = 2.0, b: float = 5.0) -> Callable[[], float]:
        """Generate a Beta distribution function."""
        def distribution():
            return np.random.beta(a=a, b=b, size=1000)
        return distribution

    def round(self, round: int) -> NDArray[np.float64]:
        return self._valuations[:, round]

    @property
    def valuations(self) -> NDArray[np.float64]:
        return self._valuations

    @property
    def time_horizon(self) -> int:
        return self._num_rounds

    @property
    def num_items(self) -> int:
        return self._num_items


if __name__ == "__main__":
    print("Stochastic Environment Test...")

    env = StochasticEnvironment(
        distribution_func=StochasticEnvironment.gaussian_distribution(),
        num_items=5,
        num_rounds=100,
        seed=42
    )

    assert env.valuations.shape == (5, 100)
    assert env.time_horizon == 100
    assert env.num_items == 5

    # from collections import Counter
    # freq = Counter([float(x) for x in env.valuations[0]])

    # from matplotlib import pyplot as plt
    # plt.bar(freq.keys(), freq.values(), width=0.01)
    # plt.show()

    print("Stochastic Environment Test Passed!")

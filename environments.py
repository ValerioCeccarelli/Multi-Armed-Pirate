

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
    def __init__(self, distribution_functions: list[Callable[[], float]], num_rounds: int, seed: int = 42):
        """
        Stochastic environment where valuations are drawn from a specified distribution.

        Args:
            distribution_functions (list[Callable[[], float]]): Functions to generate random valuations (num_items,)
            num_rounds (int): Total number of rounds.
            seed (int): Random seed for reproducibility.
        """
        self._num_items = len(distribution_functions)
        self._num_rounds = num_rounds
        self._rng = np.random.default_rng(seed)
        self._valuations = np.array(
            [[distribution_func() for _ in range(num_rounds)]
             for distribution_func in distribution_functions]
        ).astype(np.float64)

    @classmethod
    def gaussian_distribution(cls, mean: float = 0.5, std: float = 0.1) -> Callable[[], float]:
        """Generate a Gaussian distribution function."""
        def distribution():
            return np.random.normal(loc=mean, scale=std)
        return distribution

    @classmethod
    def beta_distribution(cls, a: float = 2.0, b: float = 5.0) -> Callable[[], float]:
        """Generate a Beta distribution function."""
        def distribution():
            return np.random.beta(a=a, b=b)
        return distribution

    @classmethod
    def uniform_distribution(cls, low: float = 0.0, high: float = 1.0) -> Callable[[], float]:
        """Generate a Uniform distribution function."""
        def distribution():
            return np.random.uniform(low=low, high=high)
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


class NonStochasticSmoothChangeEnvironment(Environment):
    def __init__(self, distribution_functions: list[Callable[[int], float]], num_rounds: int, seed: int = 62):
        """
        Non-stationary environment where valuations change smoothly over time.

        Args:
            distribution_functions (list[Callable[[int], float]]): Functions to generate valuations based on round (num_items,)
            num_rounds (int): Total number of rounds.
            seed (int): Random seed for reproducibility.
        """
        self._num_items = len(distribution_functions)
        self._num_rounds = num_rounds
        self._rng = np.random.default_rng(seed)
        self._valuations = np.array(
            [[distribution_func(t) for t in range(num_rounds)]
             for distribution_func in distribution_functions]
        ).astype(np.float64)

    @classmethod
    def generate_beta_valuations(cls, time_horizon: int, freq: int) -> Callable[[int], float]:
        """Generate Beta valuations with oscillating parameters"""
        def distribution(t: int) -> float:
            # Oscillating Alpha and Beta parameters
            alpha_t = 1 + 4 * \
                (0.5 + 0.5 * np.sin(freq * np.pi * t / time_horizon))
            beta_t = 1 + 4 * \
                (0.5 + 0.5 * np.cos(freq * np.pi * t / time_horizon))
            return np.random.beta(alpha_t, beta_t)
        return distribution

    @classmethod
    def generate_simple_tv(cls, time_horizon: int, num_items: int) -> Callable[[int], float]:
        """Generate simple time-varying valuations"""

        phi = 0.0
        mu0, A, f = 0.5, 0.1, 100
        sigma0, A_sigma, phi_sigma, rho0 = 0.1, 0.1, 0, 0.6

        rng = np.random.default_rng(0)
        T = time_horizon
        m = num_items
        V: np.ndarray = np.empty((T, m))
        R: np.ndarray = np.eye(m) + (1 - np.eye(m)) * rho0

        def distribution(t: int) -> float:
            mu_t = mu0 + A * np.sin(2 * np.pi * f * t / T + phi)
            sigma_t = sigma0 + A_sigma * \
                np.sin(2 * np.pi * f * t / T + phi_sigma)
            Sigma: np.ndarray = np.diag(
                [sigma_t] * m) @ R @ np.diag([sigma_t] * m)
            sample: np.ndarray = rng.multivariate_normal([mu_t] * m, Sigma)
            # print(sample.shape, sample)
            # raise NotImplementedError("Check the shape of the sample")
            return np.clip(sample[0], 0, 1)

        return distribution

    @classmethod
    def gaussian_distribution(cls, mean: float = 0.5, std: float = 0.1) -> Callable[[int], float]:
        """Generate a Gaussian distribution function that oscillates."""
        def distribution(t: int) -> float:
            return np.random.normal(loc=mean + 0.1 * np.sin(2 * np.pi * t / 100), scale=std)
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


class NonStochasticAbruptChangeEnvironment(StochasticEnvironment):
    def __init__(self, distribution_functions: list[list[Callable[[], float]]], num_rounds: int, seed: int = 42):
        """
        Non-stationary environment where valuations change abruptly at fixed intervals.

        Args:
            distribution_functions (list[list[Callable[[], float]]]): List of lists of functions to generate random valuations for each interval (num_items, num_intervals)
            num_rounds (int): Total number of rounds.
            seed (int): Random seed for reproducibility.
        """
        self._num_items = len(distribution_functions)
        assert self._num_items > 0, "There must be at least one item"
        assert all(len(
            funcs) > 0 for funcs in distribution_functions), "Each item must have at least one interval"
        self._num_intervals = len(distribution_functions[0])
        assert all(len(
            funcs) == self._num_intervals for funcs in distribution_functions), "All items must have the same number of intervals"

        self._num_rounds = num_rounds
        self._rng = np.random.default_rng(seed)

        self._valuations = np.zeros(
            (self._num_items, self._num_rounds), dtype=np.float64)

        assert num_rounds % self._num_intervals == 0, "num_rounds must be divisible by num_intervals"
        self._interval_length = num_rounds // self._num_intervals

        for t in range(num_rounds):
            interval_index = min(t // self._interval_length,
                                 self._num_intervals - 1)
            for item_index in range(self._num_items):
                self._valuations[item_index,
                                 t] = distribution_functions[item_index][interval_index]()


if __name__ == "__main__":
    print("Stochastic Environment Test...")

    env = StochasticEnvironment(
        distribution_functions=[
            StochasticEnvironment.gaussian_distribution(),
            StochasticEnvironment.gaussian_distribution(),
            StochasticEnvironment.gaussian_distribution()
        ],
        num_rounds=100,
        seed=42
    )

    assert env.valuations.shape == (3, 100)
    assert env.time_horizon == 100
    assert env.num_items == 3

    # from collections import Counter
    # freq = Counter([float(x) for x in env.valuations[0]])

    # from matplotlib import pyplot as plt
    # plt.bar(freq.keys(), freq.values(), width=0.01)
    # plt.show()

    print("Stochastic Environment Test Passed!")

    print("Non-Stochastic Smooth Change Environment Test...")
    print("Non-Stochastic Smooth Change Environment Test Passed!")

    print("Non-Stochastic Abrupt Change Environment Test ...")

    env = NonStochasticAbruptChangeEnvironment(
        distribution_functions=[
            [
                NonStochasticAbruptChangeEnvironment.uniform_distribution(
                    low=0.0, high=0.3),
                NonStochasticAbruptChangeEnvironment.uniform_distribution(
                    low=0.3, high=0.6),
                NonStochasticAbruptChangeEnvironment.uniform_distribution(
                    low=0.6, high=1.0),
            ],
            [
                NonStochasticAbruptChangeEnvironment.uniform_distribution(
                    low=0.0, high=0.3),
                NonStochasticAbruptChangeEnvironment.uniform_distribution(
                    low=0.3, high=0.6),
                NonStochasticAbruptChangeEnvironment.uniform_distribution(
                    low=0.6, high=1.0),
            ],
            [
                NonStochasticAbruptChangeEnvironment.uniform_distribution(
                    low=0.0, high=0.3),
                NonStochasticAbruptChangeEnvironment.uniform_distribution(
                    low=0.3, high=0.6),
                NonStochasticAbruptChangeEnvironment.uniform_distribution(
                    low=0.6, high=1.0),
            ],
        ],
        interval_length=50,
        num_rounds=200,
        seed=42
    )

    assert env.valuations.shape == (
        3, 200), f"Expected shape (3, 200), got {env.valuations.shape}"
    assert env.time_horizon == 200, f"Expected time_horizon 200, got {env.time_horizon}"

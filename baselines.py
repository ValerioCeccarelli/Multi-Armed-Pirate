import itertools

import numpy as np
from numpy.typing import NDArray

from agents import Agent


class FixedActionBaselineAgent:
    def __init__(
        self,
        num_items: int,
        prices: NDArray[np.float64],
        time_horizon: int,
        valuations: np.ndarray,
        budget: int = None,
    ):
        """
        Fixed action baseline agent that selects the optimal fixed price for each item
        based on perfect knowledge of future valuations.

        Args:
            num_items: Number of items
            price_set: List of possible prices for each item (num_prices,)
            time_horizon: Total number of rounds
            valuations: Known valuations for each item at each round (num_items, time_horizon)
            budget: Total budget (if None, set to time_horizon)
        """
        self.num_items = int(num_items)
        assert self.num_items > 0, "num_items must be positive"
        self.prices = list(prices)
        assert len(self.prices) > 0, "prices must be non-empty"
        self.num_prices = len(self.prices)
        if budget is None:
            budget = time_horizon * num_items
        self.initial_budget = int(budget)
        assert (
            self.initial_budget >= self.num_items
        ), "budget must be at least num_items"
        self.remaining_budget = int(budget)
        self.time_horizon = int(time_horizon)

        self.current_round = 0

        valuations = np.asarray(valuations, dtype=float)
        assert valuations.shape == (
            self.num_items,
            self.time_horizon,
        ), "valuations must be shape (num_items, time_horizon)"

        optimal_prices = self.calculate_baseline_performance(
            valuations, self.prices, self.initial_budget
        )

        self.optimal_indexes = np.array(
            [self.prices.index(p) if p in self.prices else 0 for p in optimal_prices],
            dtype=int,
        )

    def calculate_baseline_performance(
        self, valuations: np.ndarray, prices: list[float], budget: int
    ) -> list[float]:
        """
        Calculate the optimal fixed price for each item using brute-force search.

        Args:
            valuations: Known valuations for each item at each round (num_items, time_horizon)
            prices: List of possible prices for each item (num_prices,)
            budget: Total budget

        Returns:
            list of optimal prices for each item
        """
        num_items = valuations.shape[0]

        best_prices = []
        best_total_reward = -1

        # Generate all possible price combinations for items
        all_price_combinations = itertools.product(*([prices] * num_items))

        for price_combination in all_price_combinations:
            prices_array = np.array(price_combination)

            # Calculate performance for this price combination
            total_reward = self._evaluate_price_combination(
                valuations, prices_array, budget, num_items
            )
            if total_reward > best_total_reward:
                best_total_reward = total_reward
                best_prices = price_combination

        return best_prices

    def _evaluate_price_combination(
        self,
        valuations: np.ndarray,
        prices: np.ndarray,
        total_budget: int,
        num_items: int,
    ) -> float:
        """
        Evaluate a specific price combination.
        Args:
            valuations: Known valuations for each item at each round (num_items, time_horizon)
            prices: List of possible prices for each item (num_prices,)
            total_budget: Total budget
            num_items: Number of items

        Returns:
            float total reward
        """
        total_reward = 0
        remaining_budget = total_budget
        for t in range(valuations.shape[1]):
            if remaining_budget < num_items:
                break
            for item_idx in range(num_items):
                if prices[item_idx] <= valuations[item_idx][t]:
                    total_reward += prices[item_idx]
                    remaining_budget -= 1

        return total_reward

    def pull_arm(self) -> NDArray[np.int64]:
        """Return the scheduled price index for each item at current_round."""
        t = self.current_round
        if t < 0 or t >= self.time_horizon:
            raise IndexError("current_round out of bounds")

        return self.optimal_indexes

    def update(
        self, rewards: NDArray[np.float64], full_rewards: NDArray[np.float64] = None
    ) -> None:
        self.current_round += 1


class PerfectBaseline:
    def __init__(
        self,
        num_items: int,
        prices: NDArray[np.float64],
        time_horizon: int,
        valuations: np.ndarray,
        budget: int = None,
    ):
        """
        This baseline choose the prices based on the best possible reward that can be achieved.
        It's impossible that another agent performs better that this one.

        Args:
            num_items: Number of items
            prices: List of possible prices for each item (num_prices,)
            time_horizon: Total number of rounds
            valuations: Known valuations for each item at each round (num_items, time_horizon)
            budget: Total budget (if None, set to time_horizon)
        """
        self.num_items = int(num_items)
        assert self.num_items > 0, "num_items must be positive"
        assert prices.any, "prices must be non-empty"
        self.num_prices = len(prices)
        if budget is None:
            budget = time_horizon * num_items
        assert budget >= self.num_items, "budget must be at least num_items"
        self.time_horizon = int(time_horizon)
        self.current_round = 0

        valuations = np.asarray(valuations, dtype=float)
        assert valuations.shape == (
            self.num_items,
            self.time_horizon,
        ), "valuations must be shape (num_items, time_horizon)"

        triples = [
            (float(valuations[i, t]), i, t)
            for i in range(num_items)
            for t in range(time_horizon)
        ]

        # Sort descending by valuation
        triples.sort(key=lambda x: x[0], reverse=True)
        self.schedule = np.full(
            (num_items, time_horizon), self.num_prices - 1, dtype=int
        )

        # Place budget-limited selected positions
        B = min(budget, len(triples))
        for k in range(B):
            v, i, t = triples[k]
            idx = 0
            # Greatest price among the prices smaller than v
            for price_idx in range(self.num_prices):
                if prices[price_idx] <= v:
                    idx = price_idx
            self.schedule[i, t] = idx

    def pull_arm(self) -> NDArray[np.int64]:
        """Return the scheduled price index for each item at current_round."""
        t = self.current_round
        if t < 0 or t >= self.time_horizon:
            raise IndexError("current_round out of bounds")
        return self.schedule[:, t]

    def update(
        self, rewards: NDArray[np.float64], full_rewards: NDArray[np.float64] = None
    ) -> None:
        self.current_round += 1


if __name__ == "__main__":

    print("Test FixedActionBaselineAgent ...")

    num_items = 2
    price_set = np.array([0.1, 1, 10])
    time_horizon = 5

    # Create a FixedActionBaselineAgent instance
    agent = FixedActionBaselineAgent(
        num_items=num_items,
        prices=price_set,
        time_horizon=time_horizon,
        valuations=np.array([[1, 1, 1, 1, 1], [5, 5, 5, 5, 5]]),
    )

    assert tuple(agent.optimal_indexes) == (
        1,
        1,
    ), f"Expected optimal indexes (1, 1), got {agent.optimal_indexes}"

    agent = FixedActionBaselineAgent(
        num_items=num_items,
        prices=price_set,
        time_horizon=time_horizon,
        valuations=np.array([[1, 1, 1, 1, 1], [5, 5, 5, 5, 5]]),
        budget=2,
    )

    assert tuple(agent.optimal_indexes) == (
        1,
        1,
    ), f"Expected optimal indexes (1, 1), got {agent.optimal_indexes}"

    agent = FixedActionBaselineAgent(
        num_items=num_items,
        prices=price_set,
        time_horizon=time_horizon,
        valuations=np.array([[1, 10, 1, 1, 1], [5, 5, 5, 5, 5]]),
        budget=3,
    )

    assert tuple(agent.optimal_indexes) == (
        2,
        1,
    ), f"Expected optimal indexes (2, 1), got {agent.optimal_indexes}"

    agent = FixedActionBaselineAgent(
        num_items=num_items,
        prices=np.array([1, 4, 13]),
        time_horizon=4,
        valuations=np.array([[1, 1, 1, 13], [4, 4, 4, 1]]),
        budget=4,
    )

    assert tuple(agent.optimal_indexes) == (
        2,
        2,
    ), f"Expected optimal indexes (2, 2), got {agent.optimal_indexes}"

    print("FixedActionBaselineAgent created successfully.")

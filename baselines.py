
import itertools
import numpy as np
from numpy.typing import NDArray
from agents import Agent


class FixedActionBaselineAgent:
    def __init__(
        self,
        num_items: int,
        price_set: NDArray[np.float64],
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
        self.price_set = list(price_set)
        assert len(self.price_set) > 0, "price_set must be non-empty"
        self.num_prices = len(self.price_set)
        if budget is None:
            budget = time_horizon * num_items
        self.initial_budget = int(budget)
        assert self.initial_budget >= self.num_items, "budget must be at least num_items"
        self.remaining_budget = int(budget)
        self.time_horizon = int(time_horizon)

        self.current_round = 0
        self.last_chosen_price_indices = np.zeros(self.num_items, dtype=int)

        valuations = np.asarray(valuations, dtype=float)
        assert valuations.shape == (
            self.num_items,
            self.time_horizon,
        ), "valuations must be shape (num_items, time_horizon)"
        self._valuations = valuations

        optimal_prices = self.calculate_baseline_performance(
            self._valuations, self.price_set, self.initial_budget
        )

        self.optimal_indexes = np.array(
            [
                self.price_set.index(p) if p in self.price_set else 0
                for p in optimal_prices
            ],
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

        Returns:
            float total reward
        """
        # Mask where valuations are >= prices (purchases)
        purchase_mask = valuations >= prices[:, np.newaxis]

        # Rewards matrix (price * purchase)
        rewards_matrix = purchase_mask * prices[:, np.newaxis]

        # Cumulative cost over time (number of purchases)
        cumulative_purchases = np.cumsum(purchase_mask.sum(axis=0))

        # Apply budget constraint
        budget_constraint = cumulative_purchases <= (total_budget - num_items)

        # Calculate total rewards for each round
        rewards_per_round = rewards_matrix.sum(axis=0) * budget_constraint
        total_reward = rewards_per_round.sum()

        return total_reward

    def pull_arm(self) -> NDArray[np.int64]:
        """Return the scheduled price index for each item at current_round."""
        t = self.current_round
        if t < 0 or t >= self.time_horizon:
            raise IndexError("current_round out of bounds")

        indices = self.optimal_indexes
        self.last_chosen_price_indices = indices
        return indices

    def update(
        self, rewards: NDArray[np.float64], full_rewards: NDArray[np.float64] = None
    ) -> None:
        self.current_round += 1


if __name__ == "__main__":

    print("Test FixedActionBaselineAgent ...")

    num_items = 2
    price_set = np.array([0.1, 0.4, 0.7, 0.8])
    time_horizon = 5

    # Create a FixedActionBaselineAgent instance
    agent = FixedActionBaselineAgent(
        num_items=num_items,
        price_set=price_set,
        time_horizon=time_horizon,
        valuations=np.array(
            [[0.1, 0.1, 0.1, 0.1, 0.1],
             [0.5, 0.5, 0.5, 0.5, 0.5]])
    )

    assert tuple(agent.optimal_indexes) == (
        0, 1), f"Expected optimal indexes (1, 2), got {agent.optimal_indexes}"

    agent = FixedActionBaselineAgent(
        num_items=num_items,
        price_set=price_set,
        time_horizon=time_horizon,
        valuations=np.array(
            [[0.1, 0.1, 0.1, 0.1, 0.1],
             [0.5, 0.5, 0.5, 0.5, 0.5]]),
        budget=3
    )

    assert agent.optimal_indexes[
        1] == 1, f"Expected optimal index for item 2 to be 1, got {agent.optimal_indexes[1]}"
    assert agent.optimal_indexes[
        0] >= 1, f"Expected optimal index for item 1 to be >= 1, got {agent.optimal_indexes[0]}"

    agent = FixedActionBaselineAgent(
        num_items=num_items,
        price_set=price_set,
        time_horizon=time_horizon,
        valuations=np.array(
            [[0.1, 0.1, 0.1, 0.9, 0.1],
             [0.5, 0.5, 0.5, 0.5, 0.5]]),
        budget=6
    )

    assert tuple(agent.optimal_indexes) == (
        3, 1), f"Expected optimal indexes (3, 1), got {agent.optimal_indexes}"

    print("FixedActionBaselineAgent created successfully.")

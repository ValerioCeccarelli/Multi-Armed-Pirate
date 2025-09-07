from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, milp


class Agent(ABC):
    @abstractmethod
    def pull_arm(self) -> NDArray[np.int64]:
        """Select the price index (or -1 if budget depleted) for each item. (num_items,)"""
        pass

    @abstractmethod
    def update(
        self, rewards: NDArray[np.float64], valuations: NDArray[np.float64] = None
    ) -> None:
        """Update the agent's knowledge based on the results of the round.

        Args:
            rewards: Rewards received for each item. (num_items,)
            valuations: Valuations for each item. (num_items, ) Only used in full-feedback.
        """
        pass


# Task 1.1
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
                (2 * np.log(self.total_counts)) / (self.counts + 1e-5)
            )
            price_index = int(np.argmax(ucb_values))

        self._last_price_index = price_index
        return np.array([price_index], dtype=np.int64)

    def update(
        self, rewards: NDArray[np.float64], valuations: NDArray[np.float64] = None
    ) -> None:
        assert self._last_price_index != -1, "pull_arm must be called before update."

        reward = rewards[0]
        self.total_counts += 1
        self.counts[self._last_price_index] += 1
        self.values[self._last_price_index] += (
            reward - self.values[self._last_price_index]
        ) / self.counts[self._last_price_index]

        self._last_price_index = -1  # Reset for next round


# Task 1.2 & Task 2
class CombinatorialUCBBidding:
    """
    Agent for combinatorial bidding using Upper Confidence Bound (UCB).

    This agent manages price selection for multiple item types under budget constraints,
    using a combinatorial optimization approach.
    """

    def __init__(
        self,
        num_items: int,
        price_set: list[float],
        budget: int,
        time_horizon: int,
        alpha: float = 1.0,
    ):
        """
        Initialize the combinatorial UCB agent.

        Args:
            num_items: Number of item types
            price_set: Discrete set of possible prices for all items
            budget: Total initial budget
            time_horizon: Time horizon of the simulation
            alpha: UCB/LCB exploration parameter (alpha)
        """
        self.num_items = int(num_items)
        self.price_set = list(price_set)
        self.num_prices = len(self.price_set)
        self.initial_budget = int(budget)
        self.time_horizon = time_horizon
        self.exploration_param = alpha

        # Statistics stored in num_items x num_prices matrices
        self.pull_counts = np.zeros((self.num_items, self.num_prices))
        self.average_rewards = np.zeros((self.num_items, self.num_prices))
        self.average_costs = np.zeros((self.num_items, self.num_prices))

        self.remaining_budget = budget
        self.current_round = 0
        self.last_chosen_price_indices = np.zeros(self.num_items, dtype=int)

        self.schedule = np.ones(
            (self.num_items, self.time_horizon), dtype=int) * -1
        self.taken = np.zeros((self.num_items, self.time_horizon), dtype=int)

    def pull_arm(self) -> np.ndarray:
        """
        Select a price index for each of the item types.

        Returns:
            Numpy array of selected price indices.
        """
        if self.remaining_budget < self.num_items:
            # Budget depleted
            self.last_chosen_price_indices = -1 * \
                np.ones(self.num_items, dtype=int)
            return self.last_chosen_price_indices

        # Exploration phase: ensure each arm is tried at least once
        unexplored_arm = self._find_unexplored_arm()
        if unexplored_arm is not None:
            self.last_chosen_price_indices = unexplored_arm
            return unexplored_arm

        # Calculate UCB and LCB for optimization
        upper_confidence_bounds, lower_confidence_bounds = (
            self._calculate_confidence_bounds()
        )

        # Solve the combinatorial optimization problem
        chosen_indices = self._solve_optimization_problem(
            upper_confidence_bounds, lower_confidence_bounds
        )

        self.last_chosen_price_indices = chosen_indices
        return chosen_indices

    def _find_unexplored_arm(self) -> Optional[np.ndarray]:
        """
        Find the first unexplored arm.

        Returns:
            Array of price indices if unexplored arm found, None otherwise.
        """
        for item_idx in range(self.num_items):
            for price_idx in range(self.num_prices):
                if self.pull_counts[item_idx, price_idx] == 0:
                    # Explore arm (item_idx, price_idx)
                    chosen_indices = np.zeros(self.num_items, dtype=int)
                    chosen_indices[item_idx] = price_idx
                    return chosen_indices
        return None

    def _calculate_confidence_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate Upper Confidence Bounds (UCB) and Lower Confidence Bounds (LCB).

        Returns:
            Tuple of (UCB for rewards, LCB for costs)
        """
        # Add small epsilon to avoid division by zero
        confidence_term = self.exploration_param * np.sqrt(
            np.log(self.time_horizon) / (self.pull_counts + 1e-8)
        )

        reward_ucb = self.average_rewards + confidence_term
        cost_lcb = np.maximum(0, self.average_costs - confidence_term)

        return reward_ucb, cost_lcb

    def _solve_optimization_problem(
        self, reward_ucb: np.ndarray, cost_lcb: np.ndarray
    ) -> np.ndarray:
        """
        Solve the Integer Linear Programming (ILP) optimization problem.

        Args:
            reward_ucb: Upper confidence bounds for rewards
            cost_lcb: Lower confidence bounds for costs

        Returns:
            Array of optimal selected price indices
        """
        # Calculate target spending rate for the remainder of the horizon
        remaining_rounds = self.time_horizon - self.current_round
        target_spend_rate = (
            self.remaining_budget / remaining_rounds if remaining_rounds > 0 else 0
        )

        # Set up optimization problem
        objective_coefficients = -reward_ucb.flatten()  # Minimize negative = maximize

        # Constraints
        constraints = self._build_constraints(cost_lcb, target_spend_rate)

        # Binary variables
        integrality = np.ones_like(objective_coefficients)
        bounds = (0, 1)

        # Solve the ILP
        result = milp(
            c=objective_coefficients,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
        )

        if result.success and result.x is not None:
            # Extract solution
            solution_matrix = np.round(result.x).reshape(
                (self.num_items, self.num_prices)
            )
            chosen_indices = np.argmax(solution_matrix, axis=1)
        else:
            # Fallback strategy: greedy selection
            chosen_indices = self._greedy_fallback(
                reward_ucb, cost_lcb, target_spend_rate
            )

        return chosen_indices

    def _build_constraints(
        self, cost_lcb: np.ndarray, target_spend_rate: float
    ) -> list[LinearConstraint]:
        """
        Build constraints for the optimization problem.

        Args:
            cost_lcb: Lower confidence bounds for costs
            target_spend_rate: Target spending rate

        Returns:
            List of linear constraints
        """
        constraints = []

        # Constraint 1: Budget - total expected cost must not exceed spending rate
        if cost_lcb.size > 0:
            budget_constraint_matrix = np.array([cost_lcb.flatten()])
            budget_upper_bound = np.array([target_spend_rate])
            budget_lower_bound = -1e20 * np.ones_like(budget_upper_bound)
            constraints.append(
                LinearConstraint(
                    budget_constraint_matrix, budget_lower_bound, budget_upper_bound
                )
            )

        # Constraint 2: Choice - exactly one price must be chosen for each item
        choice_constraint_matrix = []
        for item_idx in range(self.num_items):
            row = np.zeros(self.num_items * self.num_prices)
            start_idx = item_idx * self.num_prices
            end_idx = (item_idx + 1) * self.num_prices
            row[start_idx:end_idx] = 1
            choice_constraint_matrix.append(row)

        if choice_constraint_matrix:
            choice_matrix = np.array(choice_constraint_matrix)
            choice_bounds = np.ones(self.num_items)
            constraints.append(
                LinearConstraint(choice_matrix, choice_bounds, choice_bounds)
            )

        return constraints

    def _greedy_fallback(
        self, reward_ucb: np.ndarray, cost_lcb: np.ndarray, target_spend_rate: float
    ) -> np.ndarray:
        """
        Greedy fallback strategy when optimizer fails.

        Args:
            reward_ucb: Upper confidence bounds for rewards
            cost_lcb: Lower confidence bounds for costs
            target_spend_rate: Target spending rate

        Returns:
            Array of selected price indices
        """
        feasible_mask = cost_lcb <= target_spend_rate
        masked_rewards = np.where(feasible_mask, reward_ucb, -np.inf)
        return np.argmax(masked_rewards, axis=1)

    def update(
        self, rewards: np.ndarray, valuations: NDArray[np.float64] = None
    ) -> None:
        """
        Update agent statistics after a round.

        Args:
            rewards: Array of rewards received for each item
        """
        chosen_price_indices = self.last_chosen_price_indices
        costs = (rewards > 0).astype(int)

        self.schedule[:, self.current_round] = [
            self.price_set[x] for x in chosen_price_indices
        ]
        self.taken[:, self.current_round] = costs

        # Update statistics for each item
        for item_idx in range(self.num_items):
            price_idx = chosen_price_indices[item_idx]

            # Increment pull count for arm (item_idx, price_idx)
            self.pull_counts[item_idx, price_idx] += 1

            # Update running averages using incremental formula: M_k = M_{k-1} + (x_k - M_{k-1}) / k
            pull_count = self.pull_counts[item_idx, price_idx]

            reward_diff = rewards[item_idx] - \
                self.average_rewards[item_idx, price_idx]
            self.average_rewards[item_idx,
                                 price_idx] += reward_diff / pull_count

            cost_diff = costs[item_idx] - \
                self.average_costs[item_idx, price_idx]
            self.average_costs[item_idx, price_idx] += cost_diff / pull_count

        assert np.all(c in (0, 1) for c in costs)
        self.remaining_budget -= np.sum(costs)
        self.current_round += 1


# Task 3
class HedgeAgent:
    def __init__(self, K: int, learning_rate: float) -> None:
        self.K = K
        self.learning_rate = learning_rate
        self.weights = np.ones(K)

    def pull_arm(self) -> int:
        x_t = self.weights / np.sum(self.weights)
        return int(np.random.choice(np.arange(self.K), p=x_t))

    def update(self, l_t: np.ndarray) -> None:
        self.weights *= np.exp(-self.learning_rate * l_t)


# Task 3
class PrimalDualAgent(Agent):
    """Primal-Dual agent with Full-Feedback for non-stationary environments"""

    def __init__(self, prices: np.ndarray, time_horizon: int, budget: float, eta: float = None) -> None:
        """
        Initialize the primal-dual pricing agent.

        Args:
            prices: List of discrete price arrays (num_prices,)
            time_horizon: Time horizon of the simulation
            budget: Total initial budget
            eta: Learning rate for dual variable update
        """
        self.prices: np.ndarray = np.array(prices)
        assert len(prices) > 1, "PrimalDualAgent requires at least 2 prices."
        self.num_prices: int = len(prices)
        self.time_horizon: int = time_horizon
        assert budget > 0, "Budget must be positive."
        self.budget: float = budget
        self.eta: float = 1.0 / np.sqrt(time_horizon) if eta is None else eta
        lr = np.sqrt(np.log(self.num_prices) / time_horizon)
        self.hedge: HedgeAgent = HedgeAgent(
            self.num_prices, lr
        )
        self.rho: float = budget / time_horizon
        self.lmbd: float = 1.0
        self.t: int = 0
        self.last_arm = -1

    def pull_arm(self) -> NDArray[np.int64]:
        if self.budget < 1:
            self.last_arm = -1
            return np.array([-1])
        self.last_arm = self.hedge.pull_arm()
        return np.array([self.last_arm])

    def update(
        self, rewards: NDArray[np.float64], valuations: NDArray[np.float64] = None
    ) -> None:
        if self.last_arm == -1:
            return

        assert rewards.shape == (1,)
        assert valuations.shape == (1,)
        v_t = valuations[0]

        sale_mask = (self.prices <= v_t).astype(float)
        f_full = self.prices * sale_mask

        L = f_full - self.lmbd * (sale_mask - self.rho)
        f_max = float(self.prices.max())
        L_up = f_max - self.lmbd * (0 - self.rho)
        L_low = 0.0 - self.lmbd * (1 - self.rho)

        losses = 1.0 - (L - L_low) / (L_up - L_low + 1e-12)
        self.hedge.update(losses)

        c_t = 1 if rewards[0] > 0 else 0
        self.budget -= c_t

        self.lmbd = np.clip(
            self.lmbd - self.eta * (self.rho - c_t), a_min=0.0, a_max=1.0 / self.rho
        )


# Task 4
class MultiItemPrimalDualAgent(Agent):
    """Primal-Dual agent with Full-Feedback for multi-product pricing"""

    def __init__(self, prices: NDArray[np.float64], time_horizon: int, budget: float, n_products: int, eta: float = None) -> None:
        """
        Initialize the multi-item dual pricing agent.
        Args:
            prices: List of discrete price arrays for each product (num_prices,)
            time_horizon: Time horizon of the simulation
            budget: Total initial budget
            eta: Learning rate for dual variable update
        """
        self.prices = prices
        assert len(
            prices) > 1, "MultiItemPrimalDualAgent requires at least 2 prices."
        self.num_prices = len(prices)
        self.time_horizon = time_horizon
        assert n_products > 0, "Number of products must be positive."
        self.n_products = n_products
        assert budget > 0, "Budget must be positive."
        self.budget = budget
        self.rho = budget / (n_products * time_horizon)
        self.eta = eta if eta is not None else 1.0 / \
            np.sqrt(time_horizon)
        lr = np.sqrt(np.log(self.num_prices) / time_horizon)
        self.hedges = [
            HedgeAgent(self.num_prices, lr) for _ in range(n_products)
        ]
        self.lmbd = 1.0
        self.last_arms = np.array([-1] * self.n_products, dtype=np.int64)

    def pull_arm(self) -> NDArray[np.int64]:
        if self.budget < 1:
            self.last_arms = np.array([-1] * self.n_products, dtype=np.int64)
            return self.last_arms

        arms = [hedge.pull_arm() for hedge in self.hedges]
        self.last_arms = np.array(arms, dtype=np.int64)
        return self.last_arms

    def update(
        self, reward: NDArray[np.float64], valuations: NDArray[np.float64] = None
    ) -> None:
        if np.all(self.last_arms == -1):
            return

        total_units_sold: int = np.sum(reward > 0)
        p_max = float(self.prices.max())
        L_up = p_max - self.lmbd * (0 - self.rho)
        L_low = 0.0 - self.lmbd * (1 - self.rho)
        norm_factor = L_up - L_low + 1e-10

        for j in range(self.n_products):
            would_sell = (self.prices <= valuations[j]).astype(float)
            f_vec = self.prices * would_sell
            L_vec = f_vec - self.lmbd * (would_sell - self.rho)
            loss_vec = 1.0 - (L_vec - L_low) / norm_factor
            loss_vec = np.clip(loss_vec, 0.0, 1.0)
            self.hedges[j].update(loss_vec)

        self.budget -= total_units_sold
        self.lmbd = float(np.clip(
            self.lmbd - self.eta *
            (self.rho * self.n_products - total_units_sold),
            a_min=0.0,
            a_max=1 / self.rho if self.rho > 0 else 1.0,
        ))


# Task 5
class CombinatorialUCBBiddingSlidingWindow(Agent):
    """
    Combinatorial UCB agent with sliding window for non-stationary environments.


    This version only considers the last 'window_size' observations for each arm
    when calculating confidence bounds, allowing better adaptation to changing environments.
    """

    def __init__(
        self,
        num_items: int,
        price_set: list[float],
        budget: int,
        time_horizon: int,
        window_size: int,
        alpha: float = 1.0,
    ):
        """
        Initialize the sliding window combinatorial UCB agent.

        Args:
            num_items: Number of item types
            price_set: Discrete set of possible prices for all items
            budget: Total initial budget
            time_horizon: Time horizon of the simulation
            window_size: Size of the sliding window (number of recent time steps to consider)
            alpha: UCB/LCB exploration parameter
        """
        self.num_items = int(num_items)
        self.price_set = list(price_set)
        self.num_prices = len(self.price_set)
        self.initial_budget = int(budget)
        self.time_horizon = time_horizon
        self.window_size = window_size
        self.exploration_param = alpha

        # Global sliding window: track last window_size rounds
        # Each element is a tuple of (chosen_price_indices, rewards, costs)
        self.action_history = deque(maxlen=window_size)

        # Current statistics based on sliding window
        self.pull_counts = np.zeros((self.num_items, self.num_prices))
        self.average_rewards = np.zeros((self.num_items, self.num_prices))
        self.average_costs = np.zeros((self.num_items, self.num_prices))

        self.remaining_budget = budget
        self.current_round = 0
        self.last_chosen_price_indices = np.zeros(self.num_items, dtype=int)

    def pull_arm(self) -> np.ndarray:
        """
        Select a price index for each of the item types using sliding window statistics.

        Returns:
            Numpy array of selected price indices.
        """
        if self.remaining_budget < self.num_items:
            # Budget depleted
            self.last_chosen_price_indices = -1 * \
                np.ones(self.num_items, dtype=int)
            return self.last_chosen_price_indices

        # Exploration phase: ensure each arm is tried at least once
        unexplored_arm = self._find_unexplored_arm()
        if unexplored_arm is not None:
            self.last_chosen_price_indices = unexplored_arm
            return unexplored_arm

        # Calculate UCB and LCB using sliding window data
        upper_confidence_bounds, lower_confidence_bounds = (
            self._calculate_confidence_bounds_sliding_window()
        )

        # Solve the combinatorial optimization problem
        chosen_indices = self._solve_optimization_problem(
            upper_confidence_bounds, lower_confidence_bounds
        )

        self.last_chosen_price_indices = chosen_indices
        return chosen_indices

    def _find_unexplored_arm(self) -> Optional[np.ndarray]:
        """
        Find the first unexplored arm (within the sliding window).

        Returns:
            Array of price indices if unexplored arm found, None otherwise.
        """
        for item_idx in range(self.num_items):
            for price_idx in range(self.num_prices):
                if self.pull_counts[item_idx, price_idx] == 0:
                    # Explore arm (item_idx, price_idx)
                    chosen_indices = np.zeros(self.num_items, dtype=int)
                    chosen_indices[item_idx] = price_idx
                    return chosen_indices
        return None

    def _calculate_confidence_bounds_sliding_window(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate UCB and LCB using only the sliding window observations.

        Returns:
            Tuple of (UCB for rewards, LCB for costs)
        """
        reward_ucb = np.zeros((self.num_items, self.num_prices))
        cost_lcb = np.zeros((self.num_items, self.num_prices))

        for item_idx in range(self.num_items):
            for price_idx in range(self.num_prices):
                n_observations = self.pull_counts[item_idx, price_idx]

                if n_observations == 0:
                    # No observations yet, use optimistic initialization
                    # Max possible reward
                    reward_ucb[item_idx, price_idx] = 1.0
                    cost_lcb[item_idx, price_idx] = 0.0  # Min possible cost
                else:
                    # Use current averages calculated from sliding window
                    mean_reward = self.average_rewards[item_idx, price_idx]
                    mean_cost = self.average_costs[item_idx, price_idx]

                    # Confidence term based on number of observations in window
                    confidence_term = self.exploration_param * np.sqrt(
                        np.log(self.current_round + 1) / n_observations
                    )

                    reward_ucb[item_idx, price_idx] = mean_reward + \
                        confidence_term
                    cost_lcb[item_idx, price_idx] = max(
                        0, mean_cost - confidence_term)

        return reward_ucb, cost_lcb

    def _solve_optimization_problem(
        self, reward_ucb: np.ndarray, cost_lcb: np.ndarray
    ) -> np.ndarray:
        """
        Solve the Integer Linear Programming (ILP) optimization problem.

        Args:
            reward_ucb: Upper confidence bounds for rewards
            cost_lcb: Lower confidence bounds for costs

        Returns:
            Array of optimal selected price indices
        """
        # Calculate target spending rate for the remainder of the horizon
        remaining_rounds = self.time_horizon - self.current_round
        target_spend_rate = (
            self.remaining_budget / remaining_rounds if remaining_rounds > 0 else 0
        )

        # Set up optimization problem
        objective_coefficients = -reward_ucb.flatten()  # Minimize negative = maximize

        # Constraints
        constraints = self._build_constraints(cost_lcb, target_spend_rate)

        # Binary variables
        integrality = np.ones_like(objective_coefficients)
        bounds = (0, 1)

        # Solve the ILP
        result = milp(
            c=objective_coefficients,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
        )

        if result.success and result.x is not None:
            # Extract solution
            solution_matrix = np.round(result.x).reshape(
                (self.num_items, self.num_prices)
            )
            chosen_indices = np.argmax(solution_matrix, axis=1)
        else:
            # Fallback strategy: greedy selection
            chosen_indices = self._greedy_fallback(
                reward_ucb, cost_lcb, target_spend_rate
            )

        return chosen_indices

    def _build_constraints(
        self, cost_lcb: np.ndarray, target_spend_rate: float
    ) -> list[LinearConstraint]:
        """
        Build constraints for the optimization problem.

        Args:
            cost_lcb: Lower confidence bounds for costs
            target_spend_rate: Target spending rate

        Returns:
            List of linear constraints
        """
        constraints = []

        # Constraint 1: Budget - total expected cost must not exceed spending rate
        if cost_lcb.size > 0:
            budget_constraint_matrix = np.array([cost_lcb.flatten()])
            budget_upper_bound = np.array([target_spend_rate])
            budget_lower_bound = -1e20 * np.ones_like(budget_upper_bound)
            constraints.append(
                LinearConstraint(
                    budget_constraint_matrix, budget_lower_bound, budget_upper_bound
                )
            )

        # Constraint 2: Choice - exactly one price must be chosen for each item
        choice_constraint_matrix = []
        for item_idx in range(self.num_items):
            row = np.zeros(self.num_items * self.num_prices)
            start_idx = item_idx * self.num_prices
            end_idx = (item_idx + 1) * self.num_prices
            row[start_idx:end_idx] = 1
            choice_constraint_matrix.append(row)

        if choice_constraint_matrix:
            choice_matrix = np.array(choice_constraint_matrix)
            choice_bounds = np.ones(self.num_items)
            constraints.append(
                LinearConstraint(choice_matrix, choice_bounds, choice_bounds)
            )

        return constraints

    def _greedy_fallback(
        self, reward_ucb: np.ndarray, cost_lcb: np.ndarray, target_spend_rate: float
    ) -> np.ndarray:
        """
        Greedy fallback strategy when optimizer fails.

        Args:
            reward_ucb: Upper confidence bounds for rewards
            cost_lcb: Lower confidence bounds for costs
            target_spend_rate: Target spending rate

        Returns:
            Array of selected price indices
        """
        feasible_mask = cost_lcb <= target_spend_rate
        masked_rewards = np.where(feasible_mask, reward_ucb, -np.inf)
        return np.argmax(masked_rewards, axis=1)

    def _recalculate_statistics_from_window(self):
        """
        Recalculate pull counts and averages from the current sliding window.
        """
        # Reset statistics
        self.pull_counts = np.zeros((self.num_items, self.num_prices))
        self.average_rewards = np.zeros((self.num_items, self.num_prices))
        self.average_costs = np.zeros((self.num_items, self.num_prices))

        # Accumulate statistics from the sliding window
        reward_sums = np.zeros((self.num_items, self.num_prices))
        cost_sums = np.zeros((self.num_items, self.num_prices))

        for chosen_price_indices, rewards, costs in self.action_history:
            for item_idx in range(self.num_items):
                price_idx = chosen_price_indices[item_idx]

                if price_idx >= 0:  # Valid price was chosen
                    self.pull_counts[item_idx, price_idx] += 1
                    reward_sums[item_idx, price_idx] += rewards[item_idx]
                    cost_sums[item_idx, price_idx] += costs[item_idx]

        # Calculate averages (avoiding division by zero)
        for item_idx in range(self.num_items):
            for price_idx in range(self.num_prices):
                count = self.pull_counts[item_idx, price_idx]
                if count > 0:
                    self.average_rewards[item_idx, price_idx] = (
                        reward_sums[item_idx, price_idx] / count
                    )
                    self.average_costs[item_idx, price_idx] = (
                        cost_sums[item_idx, price_idx] / count
                    )

    def update(
        self, rewards: np.ndarray, valuations: NDArray[np.float64] = None
    ) -> None:
        """
        Update sliding window statistics after a round.

        Args:
            rewards: Array of rewards received for each item
        """
        chosen_price_indices = self.last_chosen_price_indices
        costs = (rewards > 0).astype(int)

        # Add current round to action history
        # The deque will automatically remove the oldest round if window is full
        self.action_history.append(
            (chosen_price_indices.copy(), rewards.copy(), costs.copy())
        )

        # Recalculate statistics from the current sliding window
        self._recalculate_statistics_from_window()

        # Update budget and round counter
        assert np.all(c in (0, 1) for c in costs)
        self.remaining_budget -= np.sum(costs)
        self.current_round += 1

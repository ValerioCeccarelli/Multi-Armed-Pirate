from abc import ABC, abstractmethod
from typing import Optional, Tuple
from numpy.typing import NDArray
import numpy as np
import math
from scipy.optimize import LinearConstraint, milp


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
            self.last_chosen_price_indices = -1 * np.ones(
                self.num_items, dtype=int)
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
        self, rewards: np.ndarray, full_rewards: NDArray[np.float64] = None
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
class PrimalDualAgent(Agent):
    """
    Primal-Dual agent with Bandit Feedback for pricing.
    This agent only supports a single item (num_items=1).
    Uses EXP3.P as the primal sub-agent and updates a dual variable lambda.
    """

    def __init__(self, num_prices: int, budget: float, horizon: int, eta: float = None):
        assert num_prices > 1, "PrimalDual agent requires at least 2 prices."
        self.num_prices = num_prices
        self.budget = budget
        self.horizon = horizon
        self.eta = eta if eta is not None else 1 / math.sqrt(horizon)

        # EXP3.P as the primal agent
        self.exp3p = Exp3PAgent(K=num_prices, T=horizon, delta=0.05)

        # Budget and pacing
        self.remaining_budget = budget
        self.rho = budget / horizon  # pacing rate

        # Dual variable
        self.lmbd = 1.0

        # Tracking
        self.total_counts = 0
        self._last_price_index = -1

    def pull_arm(self) -> NDArray[np.int64]:
        if self.remaining_budget < 1:
            # Budget depleted
            self._last_price_index = -1
            return np.array([-1], dtype=np.int64)

        price_index = self.exp3p.pull_arm()
        self._last_price_index = price_index
        return np.array([price_index], dtype=np.int64)

    def update(self, rewards: NDArray[np.float64], full_rewards: NDArray[np.float64] = None) -> None:
        if self._last_price_index == -1:
            # No arm was pulled (budget depleted)
            return

        reward = rewards[0]

        # Determine if a sale occurred (cost = 1 if sale, 0 otherwise)
        # We assume reward > 0 means a sale occurred
        cost = 1.0 if reward > 0 else 0.0

        # Update budget
        self.remaining_budget -= cost
        self.total_counts += 1

        # Compute normalized reward for EXP3.P
        # The Lagrangian is: reward - lambda * (cost - rho)
        net_reward = reward - self.lmbd * (cost - self.rho)

        # Normalize to [0,1] for EXP3.P
        # Assuming price range is [0, 1], max possible reward is 1
        max_possible = 1.0 - self.lmbd * (0 - self.rho)  # when cost=0
        min_possible = 0.0 - self.lmbd * \
            (1 - self.rho)  # when cost=1, reward=0

        if max_possible > min_possible:
            normalized_reward = (net_reward - min_possible) / \
                (max_possible - min_possible)
        else:
            normalized_reward = 0.5  # fallback

        # Clamp to [0,1]
        normalized_reward = np.clip(normalized_reward, 0.0, 1.0)

        # Update EXP3.P
        self.exp3p.probs = self.exp3p._compute_probs()
        self.exp3p.update(self._last_price_index, normalized_reward)

        # Update dual variable lambda
        self.lmbd = np.clip(
            self.lmbd - self.eta * (self.rho - cost),
            a_min=0.0,
            a_max=1.0 / self.rho if self.rho > 0 else 1.0
        )

        self._last_price_index = -1  # Reset for next round


# Task 3 :)
class Exp3PAgent(Agent):
    def __init__(self, K: int, T: int, delta: float = 0.1):
        self.K = K
        self.T = T
        self.delta = delta
        self.eta = math.log(K / delta) / (T * K)
        self.gamma = 0.95 * math.log(K) / (T * K)
        self.beta = K * math.log(K) / T
        self.G = np.zeros(K)
        self.probs = np.ones(K) / K

    def _compute_probs(self) -> np.ndarray:
        expG = np.exp(self.eta * self.G)
        base = (1 - self.gamma) * (expG / expG.sum())
        probs = base + self.gamma / self.K
        return probs / probs.sum()

    def pull_arm(self) -> int:
        self.probs = self._compute_probs()
        choice = int(np.random.choice(self.K, p=self.probs))
        return choice

    def update(self, chosen: int, reward: float) -> None:
        for i in range(self.K):
            if i != chosen:
                self.G[i] += self.beta / self.probs[i]
            else:
                self.G[i] += (reward + self.beta) / self.probs[i]


# Task 3 :)
class BanditFeedbackPrimalDual(Agent):
    """Primal-Dual agent with Bandit Feedback for non-stationary pricing using EXP3.P."""

    def __init__(self, prices: np.ndarray, T: int, B: float) -> None:
        self.prices: np.ndarray = np.array(prices)
        self.K: int = len(prices)
        self.T: int = T
        self.B: float = B
        self.eta: float = 1 / np.sqrt(T)
        self.rng = np.random.default_rng()
        # Use EXP3.P as the primal (hedge) agent with a given delta
        self.exp3p: Exp3PAgent = Exp3PAgent(K=self.K, T=self.T, delta=0.05)
        self.rho: float = B / T
        self.lmbd: float = 1.0
        self.pull_counts: np.ndarray = np.zeros(self.K, int)
        self.last_arm: Optional[int] = None
        # History tracking for analysis
        self.lmbd_history: list[float] = []
        self.exp3p_weight_history: list[np.ndarray] = []

    def pull_arm(self) -> NDArray[np.int64]:
        if self.B < 1:
            self.last_arm = None
            return np.array([-1])
        self.last_arm = self.exp3p.pull_arm()
        return np.array([self.last_arm])

    def update(self, rewards: NDArray[np.float64], full_rewards: NDArray[np.float64] = None) -> None:
        if self.last_arm is None:
            return

        price_chosen = self.prices[self.last_arm]
        c_t = 1 if rewards[0] > 0 else 0  # cost is 1 if sold, 0 otherwise

        f_t = rewards[0]  # revenue is the reward received
        self.B -= c_t
        self.pull_counts[self.last_arm] += 1

        net = f_t - self.lmbd * (c_t - self.rho)

        # Normalize
        p_max = self.prices.max()
        L_up = p_max - self.lmbd * (0 - self.rho)
        L_low = 0.0 - self.lmbd * (1 - self.rho)
        norm_factor = L_up - L_low + 1e-12
        net_norm = (net - L_low) / norm_factor

        # Update the EXP3.P sub-agent using the bandit reward feedback for the chosen arm
        self.exp3p.probs = self.exp3p._compute_probs()
        self.exp3p.update(self.last_arm, net_norm)

        # Update the dual variable lambda
        self.lmbd = np.clip(self.lmbd - self.eta *
                            (self.rho - c_t), a_min=0.0, a_max=1.0 / self.rho)
        self.lmbd_history.append(self.lmbd)


# Task 4
class MultiProductPrimalDualAgent(Agent):
    """
    Primal-Dual agent with Bandit Feedback for multi-product pricing.
    Uses separate EXP3.P agents for each product and a shared dual variable lambda.
    """

    def __init__(self, num_items: int, num_prices: int, budget: float, horizon: int, eta: float = None):
        assert num_items > 0, "MultiProduct agent requires at least 1 item."
        assert num_prices > 1, "MultiProduct agent requires at least 2 prices per item."

        self.num_items = num_items
        self.num_prices = num_prices
        self.budget = budget
        self.horizon = horizon
        self.eta = eta if eta is not None else 1 / \
            math.sqrt(num_items * horizon)

        # EXP3.P agents - one for each product
        self.exp3p_agents = [
            Exp3PAgent(K=num_prices, T=horizon, delta=0.05)
            for _ in range(num_items)
        ]

        # Budget and pacing
        self.remaining_budget = budget
        self.rho = budget / (num_items * horizon)  # pacing rate per item

        # Dual variable (shared across all products)
        self.lmbd = 1.0

        # Tracking
        self.total_counts = 0
        self._last_price_indices = [-1] * num_items

    def pull_arm(self) -> NDArray[np.int64]:
        if self.remaining_budget < 1:
            # Budget depleted
            self._last_price_indices = [-1] * self.num_items
            return np.array([-1] * self.num_items, dtype=np.int64)

        # Each EXP3.P agent selects a price for its product
        price_indices = []
        for i in range(self.num_items):
            price_index = self.exp3p_agents[i].pull_arm()
            price_indices.append(price_index)

        self._last_price_indices = price_indices
        return np.array(price_indices, dtype=np.int64)

    def update(self, rewards: NDArray[np.float64], full_rewards: NDArray[np.float64] = None) -> None:
        if any(idx == -1 for idx in self._last_price_indices):
            # No arms were pulled (budget depleted)
            return

        total_revenue = 0.0
        total_sales = 0

        # Process each product
        for j in range(self.num_items):
            price_index = self._last_price_indices[j]
            reward = rewards[j]

            # Determine if a sale occurred
            cost = 1.0 if reward > 0 else 0.0

            total_revenue += reward
            total_sales += cost

            # Compute normalized reward for EXP3.P
            # The Lagrangian is: reward - lambda * (cost - rho)
            net_reward = reward - self.lmbd * (cost - self.rho)

            # Normalize to [0,1] for EXP3.P
            # Assuming max price is 1.0
            max_possible = 1.0 - self.lmbd * (0 - self.rho)  # when cost=0
            min_possible = 0.0 - self.lmbd * \
                (1 - self.rho)  # when cost=1, reward=0

            if max_possible > min_possible:
                normalized_reward = (net_reward - min_possible) / \
                    (max_possible - min_possible)
            else:
                normalized_reward = 0.5  # fallback

            # Clamp to [0,1]
            normalized_reward = np.clip(normalized_reward, 0.0, 1.0)

            # Update the corresponding EXP3.P agent
            agent = self.exp3p_agents[j]
            agent.probs = agent._compute_probs()
            agent.update(price_index, normalized_reward)

        # Update budget
        self.remaining_budget -= total_sales
        self.total_counts += 1

        # Update dual variable lambda (shared across all products)
        # The constraint is: expected total consumption <= rho * num_items
        self.lmbd = np.clip(
            self.lmbd - self.eta * (self.rho * self.num_items - total_sales),
            a_min=0.0,
            a_max=1.0 / self.rho if self.rho > 0 else 1.0
        )

        # Reset for next round
        self._last_price_indices = [-1] * self.num_items

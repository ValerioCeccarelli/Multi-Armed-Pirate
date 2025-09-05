import itertools
from scipy.optimize import milp, LinearConstraint
import numpy as np
from typing import Optional
from collections import deque
from abc import ABC, abstractmethod

class BudgetDepletedException(RuntimeError):
    """Exception raised when the agent's budget is depleted."""
    pass

class Agent(ABC):
    """
    Base class for all agents.
    """
    @abstractmethod
    def select_prices(self) -> list[float]:
        """
        Select prices for the items.
        """
        pass

    @abstractmethod
    def update(self, prices: list[float], rewards: list[float], costs: list[float]):
        """
        Update the agent's knowledge based on the observed rewards and costs.
        """
        pass
    
    @property
    @abstractmethod
    def price_set(self) -> list[float]:
        """Return the price set."""
        pass

    @property
    @abstractmethod
    def schedule(self) -> np.ndarray:
        """Return the precomputed price index schedule. (item, round)"""
        pass


class CombinatorialUCBBidding:
    """
    Agent for combinatorial bidding using Upper Confidence Bound (UCB).

    This agent manages price selection for multiple item types under budget constraints,
    using a combinatorial optimization approach.
    """

    def __init__(self, num_items: int, price_set: list[float], budget: int,
                 time_horizon: int, exploration_param: float = 1.0):
        """
        Initialize the combinatorial UCB agent.

        Args:
            num_items: Number of item types
            price_set: Discrete set of possible prices for all items
            budget: Total initial budget
            time_horizon: Time horizon of the simulation
            exploration_param: UCB/LCB exploration parameter (beta)
        """
        self.num_items = num_items
        self.price_set = price_set
        self.num_prices = len(price_set)
        self.initial_budget = budget
        self.time_horizon = time_horizon
        self.exploration_param = exploration_param

        # Statistics stored in num_items x num_prices matrices
        self.pull_counts = np.zeros((self.num_items, self.num_prices))
        self.average_rewards = np.zeros((self.num_items, self.num_prices))
        self.average_costs = np.zeros((self.num_items, self.num_prices))

        self.remaining_budget = budget
        self.current_round = 0
        self.last_chosen_price_indices = np.zeros(self.num_items, dtype=int)

    def select_prices(self) -> np.ndarray:
        """
        Select a price index for each of the item types.

        Returns:
            Numpy array of selected price indices.
        """
        # Exploration phase: ensure each arm is tried at least once
        unexplored_arm = self._find_unexplored_arm()
        if unexplored_arm is not None:
            return unexplored_arm

        # Calculate UCB and LCB for optimization
        upper_confidence_bounds, lower_confidence_bounds = self._calculate_confidence_bounds()

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

    def _solve_optimization_problem(self, reward_ucb: np.ndarray,
                                    cost_lcb: np.ndarray) -> np.ndarray:
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
        target_spend_rate = (self.remaining_budget / remaining_rounds
                             if remaining_rounds > 0 else 0)

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
                (self.num_items, self.num_prices))
            chosen_indices = np.argmax(solution_matrix, axis=1)
        else:
            # Fallback strategy: greedy selection
            chosen_indices = self._greedy_fallback(
                reward_ucb, cost_lcb, target_spend_rate)

        return chosen_indices

    def _build_constraints(self, cost_lcb: np.ndarray,
                           target_spend_rate: float) -> list[LinearConstraint]:
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
            constraints.append(LinearConstraint(
                budget_constraint_matrix, budget_lower_bound, budget_upper_bound
            ))

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
            constraints.append(LinearConstraint(
                choice_matrix, choice_bounds, choice_bounds))

        return constraints

    def _greedy_fallback(self, reward_ucb: np.ndarray, cost_lcb: np.ndarray,
                         target_spend_rate: float) -> np.ndarray:
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

    def update_statistics(self, chosen_price_indices: np.ndarray,
                          rewards: np.ndarray, costs: np.ndarray) -> None:
        """
        Update agent statistics after a round.

        Args:
            chosen_price_indices: Array of chosen price indices
            rewards: Array of rewards received for each item
            costs: Array of costs incurred for each item
        """
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

    # Compatibility methods with existing interface
    def pull_superarm(self) -> np.ndarray:
        """Alias for select_prices for compatibility."""
        return self.select_prices()

    def update(self, chosen_price_indices: np.ndarray, rewards: np.ndarray,
               costs: np.ndarray) -> None:
        """Alias for update_statistics for compatibility."""
        return self.update_statistics(chosen_price_indices, rewards, costs)


class PrimalDualAgent:
    """
    Primal-dual pacing agent with an EXP3 inner regret minimizer per item.

    This agent is adapted from req3cecc.py and provides compatibility methods
    with the existing core (select_prices/update_statistics/current_round).
    """

    def __init__(self, num_items, price_set, B, T, beta: float = 0.1, eta: float | None = None):
        """
        Args:
            num_items: number of product types (N)
            price_set: list of discrete prices shared by all items
            B: total budget (total number of items that can be sold across all types)
            T: horizon
            beta: exploration parameter for EXP3 (also used as default dual step if eta not provided)
            eta: dual step size for lambda update; if None uses beta / sqrt(T)
        """
        self.N = int(num_items)
        self.P = list(price_set)
        self.K_prices = len(self.P)

        self.T = int(T)
        self.t = 0

        self.B_initial = int(B)
        self.remaining_budget = int(B)

        # Pacing parameters
        self.rho = (B / T) if T > 0 else 0.0  # target average cost per round
        self.lambda_t = 0.0
        # Dual learning rate
        self.eta_dual = (beta / np.sqrt(max(T, 1))
                         ) if eta is None else float(eta)

        # EXP3 parameters and state (per item)
        self.gamma = float(beta)  # exploration (mixing) parameter
        # learning rate for EXP3; standard safe choice
        self.eta_exp3 = min(1.0, np.sqrt(
            np.log(max(self.K_prices, 2)) / (self.K_prices * max(self.T, 1))))
        self.weights = np.ones((self.N, self.K_prices), dtype=float)
        self.last_probs = np.full(
            (self.N, self.K_prices), 1.0 / max(self.K_prices, 1), dtype=float)
        # Stores the indices chosen in the last call to pull_superarm
        self.last_chosen_indices = np.zeros(self.N, dtype=int)

        # Compatibility attribute with core
        self.current_round = 0

        # RNG
        self._rng = np.random.default_rng()

        self.schedule = np.ones((self.N, self.T), dtype=int) * -1

    @property
    def price_set(self) -> list[float]:
        """Return the price set."""
        return self.P

    def _distributions(self) -> np.ndarray:
        """Return current action distributions per item, with exploration mixing."""
        # Normalize weights per item
        w_sum = self.weights.sum(axis=1, keepdims=True)
        # Avoid division by zero
        w_sum[w_sum == 0] = 1.0
        p = (1.0 - self.gamma) * (self.weights / w_sum) + \
            self.gamma / self.K_prices
        # numerical safety
        p = np.clip(p, 1e-12, 1.0)
        # renormalize rows
        p /= p.sum(axis=1, keepdims=True)
        return p

    def select_prices(self) -> np.ndarray:
        """
        Sample one price index for each item according to EXP3 distributions.
        Returns: np.ndarray shape (N,) of indices in [0, K_prices).
        """
        if self.remaining_budget <= 0:
            raise BudgetDepletedException("No remaining budget to select prices.")

        probs = self._distributions()
        self.last_probs = probs
        indices = np.array([
            self._rng.choice(self.K_prices, p=probs[i]) for i in range(self.N)
        ], dtype=int)
        self.last_chosen_indices = indices
        return indices

    def update(self, chosen_price_indices: np.ndarray, rewards: np.ndarray, costs: np.ndarray):
        """
        Update EXP3 weights using Lagrangian gains and update the dual variable.

        Args:
            chosen_price_indices: indices chosen per item (shape (N,))
            rewards: realized revenue per item (shape (N,))
            costs: realized cost per item (0/1) (shape (N,))
        """

        self.schedule[:, self.t] = chosen_price_indices

        # Sanity cast
        chosen_price_indices = np.asarray(chosen_price_indices, dtype=int)
        rewards = np.asarray(rewards, dtype=float)
        costs = np.asarray(costs, dtype=float)

        # Lagrangian gain per item for the chosen arm
        gains = rewards - self.lambda_t * costs  # shape (N,)

        # Update EXP3 weights using importance-weighted gain estimates
        for i in range(self.N):
            k = int(chosen_price_indices[i])
            # Guard index
            if k < 0 or k >= self.K_prices:
                continue
            p_ik = float(self.last_probs[i, k]
                         ) if self.last_probs.size else 0.0
            if p_ik <= 0:
                continue
            # Importance-weighted unbiased estimate of gain
            ghat = gains[i] / p_ik
            # EXP3 multiplicative update
            self.weights[i, k] *= np.exp(self.eta_exp3 *
                                         ghat / max(self.K_prices, 1))

        # Budget accounting
        consumed = float(costs.sum())
        self.remaining_budget -= int(consumed)

        # Dual update with projection to [0, 1/rho]
        upper = (1.0 / self.rho) if self.rho > 0 else 1e6
        self.lambda_t = float(
            np.clip(self.lambda_t - self.eta_dual *
                    (self.rho - consumed), 0.0, upper)
        )

        self.t += 1
        # Keep compatibility attribute in sync
        self.current_round = self.t

    def update_statistics(self, chosen_price_indices: np.ndarray,
                          rewards: np.ndarray, costs: np.ndarray) -> None:
        """Alias for update for compatibility with core."""
        self.update(chosen_price_indices, rewards, costs)


class SlidingWindowUCBBidding:
    """
    Agent for combinatorial bidding using Upper Confidence Bound (UCB) with sliding window.

    This agent maintains only the last T observations for each (item, price) pair,
    forgetting older observations. Statistics are computed only on the sliding window.
    """

    def __init__(self, num_items: int, price_set: list[float], budget: int,
                 time_horizon: int, window_size: int, exploration_param: float = 1.0):
        """
        Initialize the sliding window UCB agent.

        Args:
            num_items: Number of item types
            price_set: Discrete set of possible prices for all items
            budget: Total initial budget
            time_horizon: Time horizon of the simulation
            window_size: Size of the sliding window (T)
            exploration_param: UCB/LCB exploration parameter (beta)
        """
        self.num_items = num_items
        self.price_set = price_set
        self.num_prices = len(price_set)
        self.initial_budget = budget
        self.time_horizon = time_horizon
        self.window_size = window_size
        self.exploration_param = exploration_param

        # Sliding window storage: each cell contains a deque of (reward, cost) tuples
        self.sliding_windows = [[deque(maxlen=window_size) for _ in range(self.num_prices)]
                                for _ in range(self.num_items)]

        # Cache for statistics to avoid recomputing every time
        self.average_rewards = np.zeros((self.num_items, self.num_prices))
        self.average_costs = np.zeros((self.num_items, self.num_prices))
        self.pull_counts = np.zeros((self.num_items, self.num_prices))

        self.remaining_budget = budget
        self.current_round = 0
        self.last_chosen_price_indices = np.zeros(self.num_items, dtype=int)

    def select_prices(self) -> np.ndarray:
        """
        Select a price index for each of the item types.

        Returns:
            Numpy array of selected price indices.
        """
        # Exploration phase: ensure each arm is tried at least once
        unexplored_arm = self._find_unexplored_arm()
        if unexplored_arm is not None:
            return unexplored_arm

        # Calculate UCB and LCB for optimization
        upper_confidence_bounds, lower_confidence_bounds = self._calculate_confidence_bounds()

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
                if len(self.sliding_windows[item_idx][price_idx]) == 0:
                    # Explore arm (item_idx, price_idx)
                    chosen_indices = np.zeros(self.num_items, dtype=int)
                    chosen_indices[item_idx] = price_idx
                    return chosen_indices
        return None

    def _calculate_confidence_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate Upper Confidence Bounds (UCB) and Lower Confidence Bounds (LCB)
        based only on the sliding window data.

        Returns:
            Tuple of (UCB for rewards, LCB for costs)
        """
        # Update cached statistics from sliding windows
        self._update_cached_statistics()

        # Calculate confidence bounds using Hoeffding's inequality
        # For sliding window, use the actual number of samples in the window
        confidence_term = self.exploration_param * np.sqrt(
            np.log(self.time_horizon) / (self.pull_counts + 1e-8)
        )

        reward_ucb = self.average_rewards + confidence_term
        cost_lcb = np.maximum(0, self.average_costs - confidence_term)

        return reward_ucb, cost_lcb

    def _update_cached_statistics(self) -> None:
        """
        Update cached statistics based on current sliding window content.
        """
        for item_idx in range(self.num_items):
            for price_idx in range(self.num_prices):
                window = self.sliding_windows[item_idx][price_idx]
                window_size = len(window)

                if window_size > 0:
                    rewards, costs = zip(*window)
                    self.average_rewards[item_idx,
                                         price_idx] = np.mean(rewards)
                    self.average_costs[item_idx, price_idx] = np.mean(costs)
                    self.pull_counts[item_idx, price_idx] = window_size
                else:
                    self.average_rewards[item_idx, price_idx] = 0.0
                    self.average_costs[item_idx, price_idx] = 0.0
                    self.pull_counts[item_idx, price_idx] = 0

    def _solve_optimization_problem(self, reward_ucb: np.ndarray,
                                    cost_lcb: np.ndarray) -> np.ndarray:
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
        target_spend_rate = (self.remaining_budget / remaining_rounds
                             if remaining_rounds > 0 else 0)

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
                (self.num_items, self.num_prices))
            chosen_indices = np.argmax(solution_matrix, axis=1)
        else:
            # Fallback strategy: greedy selection
            chosen_indices = self._greedy_fallback(
                reward_ucb, cost_lcb, target_spend_rate)

        return chosen_indices

    def _build_constraints(self, cost_lcb: np.ndarray,
                           target_spend_rate: float) -> list[LinearConstraint]:
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
            constraints.append(LinearConstraint(
                budget_constraint_matrix, budget_lower_bound, budget_upper_bound
            ))

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
            constraints.append(LinearConstraint(
                choice_matrix, choice_bounds, choice_bounds))

        return constraints

    def _greedy_fallback(self, reward_ucb: np.ndarray, cost_lcb: np.ndarray,
                         target_spend_rate: float) -> np.ndarray:
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

    def update_statistics(self, chosen_price_indices: np.ndarray,
                          rewards: np.ndarray, costs: np.ndarray) -> None:
        """
        Update agent statistics after a round using sliding window approach.

        Args:
            chosen_price_indices: Array of chosen price indices
            rewards: Array of rewards received for each item
            costs: Array of costs incurred for each item
        """
        # Update sliding windows for each item
        for item_idx in range(self.num_items):
            price_idx = chosen_price_indices[item_idx]

            # Add new observation to the sliding window
            # If window is full, oldest observation will be automatically removed
            self.sliding_windows[item_idx][price_idx].append(
                (rewards[item_idx], costs[item_idx])
            )

        # Update budget and round counter
        assert np.all(c in (0, 1) for c in costs)
        self.remaining_budget -= np.sum(costs)
        self.current_round += 1

    # Compatibility methods with existing interface
    def pull_superarm(self) -> np.ndarray:
        """Alias for select_prices for compatibility."""
        return self.select_prices()

    def update(self, chosen_price_indices: np.ndarray, rewards: np.ndarray,
               costs: np.ndarray) -> None:
        """Alias for update_statistics for compatibility."""
        return self.update_statistics(chosen_price_indices, rewards, costs)


class FixedActionBaselineAgent:
    def __init__(self, num_items: int, price_set: list[float], budget: int,
                 time_horizon: int, valuations: np.ndarray):
        """
        Args:
            num_items: number of item types (N)
            price_set: discrete, shared price set across items (arms)
            budget: total number of allowed purchases across horizon (B)
            time_horizon: number of rounds (T)
            valuations: matrix shape (N, T) with valuation for (item, round)
        """
        self.num_items = int(num_items)
        self.price_set = list(price_set)
        self.num_prices = len(self.price_set)
        self.initial_budget = int(budget)
        self.remaining_budget = int(budget)
        self.time_horizon = int(time_horizon)

        self.current_round = 0
        self.last_chosen_price_indices = np.zeros(self.num_items, dtype=int)

        valuations = np.asarray(valuations, dtype=float)
        assert valuations.shape == (self.num_items, self.time_horizon), "valuations must be shape (num_items, time_horizon)"
        self._valuations = valuations

        optimal_prices = self.calculate_baseline_performance(
            self._valuations, self.price_set, self.initial_budget)

        self.optimal_indexes = np.array([
            self.price_set.index(p) if p in self.price_set else 0 for p in optimal_prices
        ], dtype=int)
                

    def calculate_baseline_performance(self, valuations: np.ndarray, prices: list[float], budget: int) -> list[float]:
        """
        Calculate optimal baseline performance for the given environment.
        
        The baseline represents the best possible fixed pricing strategy,
        calculated with perfect knowledge of future valuations.
        
        Args:
            env: Simulation environment
            
        Returns:
            list of optimal prices for each item
        """
        num_items = valuations.shape[0]

        best_prices = []
        best_total_reward = -1

        # Generate all possible price combinations for items
        all_price_combinations = list(itertools.product(*([prices]*num_items)))

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
    
    def _evaluate_price_combination(self, valuations: np.ndarray, prices: np.ndarray, 
                               total_budget: int, num_items: int) -> float:
        """
        Evaluate a specific price combination.
        
        Args:
            env: Simulation environment
            prices: Array of prices for each item
            total_budget: Total available budget
            num_items: Number of items
            
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

    def select_prices(self) -> np.ndarray:
        """Return the scheduled price index for each item at current_round."""
        t = self.current_round
        if t < 0 or t >= self.time_horizon:
            raise IndexError("current_round out of bounds")
        
        indices = self.optimal_indexes
        self.last_chosen_price_indices = indices
        return indices
    
    @property
    def schedule(self) -> np.ndarray:
        """Return the fixed optimal price index schedule. (item, round)"""
        return np.tile(self.optimal_indexes[:, np.newaxis], (1, self.time_horizon))

    # Compatibility alias
    def update(self, chosen_price_indices: np.ndarray, rewards: np.ndarray,
               costs: np.ndarray) -> None:
        self.current_round += 1

class IntervalAwareBaselineAgent:
    """
    Baseline agent that precomputes an interval-aware price schedule using
    full knowledge of (item, round) valuations and a global budget.

    Logic adapted from req4_baseline.py:
    - Rank all (item, round) by valuation descending.
    - Select the top B positions (B = budget).
    - For selected positions, price at the highest arm not exceeding the valuation.
    - For all other positions, price at the maximum arm (discourage purchase).

    Interfaces match other agents in this module: select_prices/pull_superarm,
    update/update_statistics, current_round, remaining_budget, last_chosen_price_indices.
    """

    def __init__(self, num_items: int, price_set: list[float], budget: int,
                 time_horizon: int, valuations: np.ndarray):
        """
        Args:
            num_items: number of item types (N)
            price_set: discrete, shared price set across items (arms)
            budget: total number of allowed purchases across horizon (B)
            time_horizon: number of rounds (T)
            valuations: matrix shape (N, T) with valuation for (item, round)
        """
        self.num_items = int(num_items)
        self.price_set = list(price_set)
        self.num_prices = len(self.price_set)
        self.initial_budget = int(budget)
        self.remaining_budget = int(budget)
        self.time_horizon = int(time_horizon)

        self.current_round = 0
        self.last_chosen_price_indices = np.zeros(self.num_items, dtype=int)

        valuations = np.asarray(valuations, dtype=float)
        assert valuations.shape == (self.num_items, self.time_horizon), "valuations must be shape (num_items, time_horizon)"
        self._valuations = valuations

        # Precompute the schedule of price indices per (item, round)
        self._schedule = self._build_schedule()

    @property
    def schedule(self) -> np.ndarray:
        """Return the precomputed price index schedule. (item, round)"""
        return self._schedule

    def _build_schedule(self) -> np.ndarray:
        # Build list of (valuation, item, round)
        n_items, n_rounds = self._valuations.shape
        triples = [(float(self._valuations[i, t]), i, t)
                   for i in range(n_items)
                   for t in range(n_rounds)]
        # Sort descending by valuation
        triples.sort(key=lambda x: x[0], reverse=True)

        # Initialize with max price index to discourage purchases by default
        max_idx = self.num_prices - 1 if self.num_prices > 0 else 0
        schedule = np.full((n_items, n_rounds), max_idx, dtype=int)

        # Place budget-limited selected positions
        B = min(self.initial_budget, len(triples))
        price_array = np.array(self.price_set, dtype=float)
        for k in range(B):
            v, i, t = triples[k]
            # Choose the index of the largest price <= v (no sorting assumption)
            mask = price_array <= (v + 1e-9)
            if mask.any():
                idx = int(np.argmax(np.where(mask, price_array, -np.inf)))
            else:
                # If all prices exceed v, pick the smallest price
                idx = int(np.argmin(price_array))
            schedule[i, t] = idx
        return schedule

    def select_prices(self) -> np.ndarray:
        """Return the scheduled price index for each item at current_round."""
        t = self.current_round
        if t < 0 or t >= self.time_horizon:
            raise IndexError("current_round out of bounds")
        
        indices = self._schedule[:, t].astype(int)
        self.last_chosen_price_indices = indices
        return indices

    # Compatibility alias
    def update(self, chosen_price_indices: np.ndarray, rewards: np.ndarray,
               costs: np.ndarray) -> None:
        self.current_round += 1


if __name__ == "__main__":
    from environment import AbruptSlightlyNonstationaryEnvironment

    env = AbruptSlightlyNonstationaryEnvironment(
        mean=[[10, 50], [100, 10], [1, 1]],  # Valuation means for each item
        std=[[2, 2], [2, 2], [2, 2]],      # Standard deviations for each item
        time_horizon=30
    )

    agent = IntervalAwareBaselineAgent(
        num_items=3,
        price_set=list(range(1, 100, 5)),
        budget=32,
        time_horizon=30,
        valuations=env.valuations
    )

    print(agent._schedule)

    for t in range(env.time_horizon):
        prices_idx = agent.select_prices()
        assert len(prices_idx) == 3
        prices = [agent.price_set[idx] for idx in prices_idx]
        l = env.round(t, prices)
        agent.update(None, None, None)
        print(f"Round {t}: ind {prices_idx} Prices {prices}, Rewards {l}")
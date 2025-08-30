from scipy.optimize import milp, LinearConstraint
import numpy as np
from typing import Optional


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
            solution_matrix = np.round(result.x).reshape((self.num_items, self.num_prices))
            chosen_indices = np.argmax(solution_matrix, axis=1)
        else:
            # Fallback strategy: greedy selection
            chosen_indices = self._greedy_fallback(reward_ucb, cost_lcb, target_spend_rate)

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
            constraints.append(LinearConstraint(choice_matrix, choice_bounds, choice_bounds))
        
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
            
            reward_diff = rewards[item_idx] - self.average_rewards[item_idx, price_idx]
            self.average_rewards[item_idx, price_idx] += reward_diff / pull_count
            
            cost_diff = costs[item_idx] - self.average_costs[item_idx, price_idx]
            self.average_costs[item_idx, price_idx] += cost_diff / pull_count

        assert np.any(c not in (0, 1) for c in costs)
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
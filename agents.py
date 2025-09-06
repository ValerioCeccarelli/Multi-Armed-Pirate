from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np
import math


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


class Exp3PAgent:
    """
    EXP3.P agent for adversarial bandits with high-probability regret bounds.
    Simplified version for use as a sub-agent in PrimalDualAgent.
    """

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
        min_possible = 0.0 - self.lmbd * (1 - self.rho)  # when cost=1, reward=0
        
        if max_possible > min_possible:
            normalized_reward = (net_reward - min_possible) / (max_possible - min_possible)
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
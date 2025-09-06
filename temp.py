from abc import ABC, abstractmethod
from dataclasses import dataclass

from numpy.typing import NDArray
import numpy as np
from matplotlib import pyplot as plt


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


@dataclass
class RunSimulationResult:
    """
    Result of a single simulation run.

    Attributes:
        valuations: Valuations matrix (num_items, time_horizon)
        played_arms: Played arms matrix (num_items, time_horizon) (-1 if not played else arm_index)
    """
    valuations: NDArray[np.float64]  # (num_items, time_horizon)
    played_arms: NDArray[np.int64]  # (num_items, time_horizon) (-1/int)


def run_simulation(env: Environment, agent: Agent, prices: NDArray[np.float64]) -> RunSimulationResult:
    """
    Run a simulation of the agent interacting with the environment.

    Args:
        env: The environment instance.
        agent: The agent instance.
        prices: Prices array (num_prices,)

    Returns:
        RunSimulationResult: The result of the simulation containing valuations and played arms.
    """

    time_horizon = env.time_horizon
    num_items = env.num_items

    total_played_arms = np.full((num_items, time_horizon), -1, dtype=np.int64)

    for t in range(time_horizon):
        # Agent interaction
        price_indexes = agent.pull_arm()
        assert price_indexes.shape == (
            env.num_items,), f"Expected shape {(env.num_items,)}, got {price_indexes.shape}"

        # Environment interaction
        valuations = env.round(t)
        assert valuations.shape == (
            env.num_items,), f"Expected shape {(env.num_items,)}, got {valuations.shape}"

        # Determine purchases and rewards
        rewards = np.zeros(num_items, dtype=np.float64)
        is_budget_exhausted = False
        for i in range(num_items):
            if price_indexes[i] == -1:
                assert np.all(
                    price_indexes == -1), f"Once an item has budget exhausted (-1), all subsequent items must also be -1. Instead got {price_indexes}"
                is_budget_exhausted = True
                break  # Budget exhausted for this item, from now on -1 by default

            price = prices[price_indexes[i]]
            if valuations[i] >= price:
                rewards[i] = price

        if is_budget_exhausted:
            break  # Stop simulation if any item's budget is exhausted

        # Update agent with rewards
        agent.update(rewards)

        total_played_arms[:, t] = price_indexes

    return RunSimulationResult(valuations=env.valuations, played_arms=total_played_arms)


def plot_cumulative_regret(
        valuations: NDArray[np.float64],
        agents_played_arms: NDArray[np.int64],
        baseline_played_arms: NDArray[np.int64],
        prices: NDArray[np.float64],
        agents_names: list[str],  # (num_agents,)
        title: str = "Cumulative Regret Over Time",
        ax: plt.Axes = None) -> None:
    """
    Plot cumulative regret over time.

    Args:
        valuations: Valuations matrix (num_trials, num_items, time_horizon)
        agents_played_arms: Played arms by agents (num_agents, num_trials, num_items, time_horizon)
        baseline_played_arms: Played arms by baseline (num_trials, num_items, time_horizon)
        prices: Prices matrix (num_prices,)
        agents_names: Names of the agents (num_agents,)
        title: Title of the plot
        ax: Matplotlib Axes to plot on. If None, creates a new figure and axes and plots there.
    """

    num_agents, num_trials, num_items, time_horizon = agents_played_arms.shape
    assert valuations.shape == (
        num_trials, num_items, time_horizon), f"Expected valuations shape {(num_trials, num_items, time_horizon)}, got {valuations.shape}"
    assert baseline_played_arms.shape == (
        num_trials, num_items, time_horizon), f"Expected baseline_played_arms shape {(num_trials, num_items, time_horizon)}, got {baseline_played_arms.shape}"
    assert len(
        agents_names) == num_agents, f"Expected {num_agents} agent names, got {len(agents_names)}"

    is_new_figure = ax is None
    if is_new_figure:
        fig, ax = plt.subplots(figsize=(10, 6))

    for agent_idx in range(num_agents):
        # (num_trials, num_items, time_horizon)
        agent_played_prices = prices[agents_played_arms[agent_idx]]
        # (num_trials, num_items, time_horizon)
        baseline_played_prices = prices[baseline_played_arms]

        # (num_trials, num_items, time_horizon)
        agent_rewards = np.where(
            valuations >= agent_played_prices, agent_played_prices, 0.0)
        # (num_trials, num_items, time_horizon)
        baseline_rewards = np.where(
            valuations >= baseline_played_prices, baseline_played_prices, 0.0)

        average_agent_rewards = np.mean(
            np.sum(agent_rewards, axis=1), axis=0)  # (time_horizon,)
        average_baseline_rewards = np.mean(
            np.sum(baseline_rewards, axis=1), axis=0)  # (time_horizon,)

        std_agent_rewards = np.std(
            np.sum(agent_rewards, axis=1), axis=0)  # (time_horizon,)
        std_baseline_rewards = np.std(
            np.sum(baseline_rewards, axis=1), axis=0)  # (time_horizon,)

        cumulative_regrets = np.cumsum(
            average_baseline_rewards - average_agent_rewards)  # (time_horizon,)
        cumulative_std = np.sqrt(
            np.cumsum(std_agent_rewards**2 + std_baseline_rewards**2))  # (time_horizon,)

        ax.plot(cumulative_regrets, label=agents_names[agent_idx])
        ax.fill_between(np.arange(time_horizon), cumulative_regrets - cumulative_std, cumulative_regrets +
                        cumulative_std, alpha=0.3, label=f"{agents_names[agent_idx]} confidence interval")

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Regret")
    ax.legend()

    if is_new_figure:
        plt.show()


class RandomAgent(Agent):
    def __init__(self, num_items: int, num_prices: int, seed: int | None = None) -> None:
        self.num_items = num_items
        self.num_prices = num_prices
        self.random_state = np.random.default_rng(seed)

    def pull_arm(self) -> NDArray[np.int64]:
        return self.random_state.integers(0, self.num_prices, size=self.num_items, dtype=np.int64)

    def update(self, rewards: NDArray[np.float64]) -> None:
        pass  # Random agent does not learn


class RandomEnvironment(Environment):
    def __init__(self, prices: NDArray[np.float64], num_items: int, time_horizon: int) -> None:
        """
        Environment with fixed valuations over time.

        Args:
            prices: Prices array (num_prices,)
            num_items: Number of items
            time_horizon: Time horizon for the environment
        """
        self._valuations = self.generate_random_valuations(
            num_items, time_horizon)
        self._prices = prices

    @staticmethod
    def generate_random_valuations(num_items: int, time_horizon: int, low: float = 0.0, high: float = 1.0, seed: int | None = None) -> NDArray[np.float64]:
        random_state = np.random.default_rng(seed)
        return random_state.uniform(low, high, size=(num_items, time_horizon))

    def round(self, round: int) -> NDArray[np.float64]:
        return self._valuations[:, round]

    @property
    def valuations(self) -> NDArray[np.float64]:
        return self._valuations

    @property
    def time_horizon(self) -> int:
        return self._valuations.shape[1]

    @property
    def num_items(self) -> int:
        return self._valuations.shape[0]


if __name__ == "__main__":
    # Example usage with multiple trials
    num_items = 5
    num_prices = 3
    time_horizon = 100
    num_trials = 10
    prices = np.array([0.2, 0.5, 0.8])

    # Preallocate containers
    # valuations: (num_trials, num_items, time_horizon)
    valuations = np.empty(
        (num_trials, num_items, time_horizon), dtype=np.float64)
    # agents_played_arms: (num_agents=1, num_trials, num_items, time_horizon)
    agents_played_arms = np.empty(
        (1, num_trials, num_items, time_horizon), dtype=np.int64)
    # baseline_played_arms: (num_trials, num_items, time_horizon)
    baseline_played_arms = np.empty(
        (num_trials, num_items, time_horizon), dtype=np.int64)

    for trial in range(num_trials):
        # New environment each trial to vary valuations across trials
        env = RandomEnvironment(
            prices=prices, num_items=num_items, time_horizon=time_horizon)

        # Agent and baseline for this trial (different seeds for reproducibility)
        agent = RandomAgent(num_items=num_items,
                            num_prices=num_prices, seed=42 + trial)
        baseline_agent = RandomAgent(
            num_items=num_items, num_prices=num_prices, seed=24 + trial)

        result = run_simulation(env, agent, prices)
        baseline_results = run_simulation(env, baseline_agent, prices)

        valuations[trial] = result.valuations  # (num_items, time_horizon)
        # (num_items, time_horizon)
        agents_played_arms[0, trial] = result.played_arms
        # (num_items, time_horizon)
        baseline_played_arms[trial] = baseline_results.played_arms

    # Plot cumulative regret averaged across trials
    plot_cumulative_regret(
        valuations=valuations,
        agents_played_arms=agents_played_arms,
        baseline_played_arms=baseline_played_arms,
        prices=prices,
        agents_names=["Random Agent"]
    )

    print("Simulation complete.")

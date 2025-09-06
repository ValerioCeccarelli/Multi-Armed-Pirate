

from dataclasses import dataclass
from typing import Callable, TypeVar
from numpy.typing import NDArray
import numpy as np
from matplotlib import pyplot as plt

from environments import Environment, StochasticEnvironment
from agents import Agent, UCBAgent
from baselines import FixedActionBaselineAgent
from plotting import plot_price_frequency_histograms


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


@dataclass
class RunMultipleSimulationsResult:
    """
    Result of multiple simulation runs.

    Attributes:
        valuations: Valuations tensor (num_trials, num_items, time_horizon)
        played_arms: Played arms tensor (num_trials, num_items, time_horizon) (-1 if not played else arm_index)
    """
    valuations: NDArray[np.float64]  # (num_trials, num_items, time_horizon)
    # (num_trials, num_items, time_horizon) (-1/int)
    agent_played_arms: NDArray[np.int64]
    baseline_played_arms: NDArray[np.int64]


AgentConfigType = TypeVar('AgentConfigType')
BaselineConfigType = TypeVar('BaselineConfigType')


def run_multiple_simulations(
        env_builder: Callable[[], Environment],
        agent_builder: Callable[[AgentConfigType], Agent],
        baseline_builder: Callable[[BaselineConfigType, Environment], Agent],
        num_trials: int,
        agent_config: AgentConfigType,
        baseline_config: BaselineConfigType,
        prices: NDArray[np.float64]) -> RunMultipleSimulationsResult:
    """
    Run multiple simulations of the agent interacting with the environment.

    Args:
        env_builder: A callable that returns a new Environment instance.
        agent_builder: A callable that takes agent_configs and returns a new Agent instance.
        baseline_builder: A callable that takes baseline_config and returns a new baseline Agent instance.
        num_trials: Number of simulation trials to run.
        agent_configs: Configuration object to pass to the agent_builder.
        baseline_config: Configuration object to pass to the baseline_builder.

    Returns:
        RunMultipleSimulationsResult: The result of the simulations containing valuations and played arms.
    """

    # Initialize environment and agent to get dimensions
    temp_env = env_builder()
    num_items = temp_env.num_items
    time_horizon = temp_env.time_horizon

    valuations = np.zeros(
        (num_trials, num_items, time_horizon), dtype=np.float64)

    agent_played_arms = np.full(
        (num_trials, num_items, time_horizon), -1, dtype=np.int64)
    baseline_played_arms = np.full(
        (num_trials, num_items, time_horizon), -1, dtype=np.int64)

    for trial in range(num_trials):
        env = env_builder()
        agent = agent_builder(agent_config)
        baseline_agent = baseline_builder(baseline_config, env)

        agent_result = run_simulation(env, agent, prices)
        baseline_results = run_simulation(env, baseline_agent, prices)

        # (num_items, time_horizon)
        valuations[trial] = agent_result.valuations
        # (num_items, time_horizon)
        agent_played_arms[trial] = agent_result.played_arms
        baseline_played_arms[trial] = baseline_results.played_arms

    return RunMultipleSimulationsResult(
        valuations=valuations,
        agent_played_arms=agent_played_arms,
        baseline_played_arms=baseline_played_arms
    )


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


num_trials = 10
time_horizon = 1000
num_prices = 10
prices = np.linspace(0.1, 1, num_prices, dtype=np.float64)


def env_builder() -> Environment:
    return StochasticEnvironment(
        distribution_functions=[
            StochasticEnvironment.gaussian_distribution(mean=0.5, std=0.1)],
        num_rounds=time_horizon,
    )


@dataclass
class UCBAgentConfig:
    num_prices: int
    alpha: float = 1.0


def agent_builder(config: UCBAgentConfig) -> Agent:
    assert isinstance(
        config, UCBAgentConfig), f"Expected UCBAgentConfig, got {type(config)}"
    return UCBAgent(
        num_prices=config.num_prices,
        alpha=config.alpha
    )


@dataclass
class BaselineAgentConfig:
    pass


def baseline_builder(config: BaselineAgentConfig, env: Environment) -> Agent:
    assert isinstance(
        config, BaselineAgentConfig), f"Expected BaselineAgentConfig, got {type(config)}"
    return FixedActionBaselineAgent(
        num_items=env.num_items,
        price_set=prices,
        time_horizon=time_horizon,
        valuations=env.valuations
    )


results = run_multiple_simulations(
    env_builder=env_builder,
    agent_builder=agent_builder,
    baseline_builder=baseline_builder,
    num_trials=num_trials,
    agent_config=UCBAgentConfig(num_prices=num_prices),
    baseline_config=BaselineAgentConfig(),
    prices=prices
)

plot_cumulative_regret(
    valuations=results.valuations,
    agents_played_arms=results.agent_played_arms[np.newaxis, ...],
    baseline_played_arms=results.baseline_played_arms,
    prices=prices,
    agents_names=["UCB Agent"],
    title="Cumulative Regret of UCB Agent vs Random Baseline"
)

plot_price_frequency_histograms(
    valuations=results.valuations,
    agents_played_arms=results.agent_played_arms[np.newaxis, ...],
    prices=prices,
    agents_names=["UCB Agent"],
)

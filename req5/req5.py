import os
import sys
from dataclasses import dataclass
from typing import Callable, TypeVar

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from agents import Agent, CombinatorialUCBBiddingSlidingWindow
from baselines import (
    FixedActionBaselineAgent,
    OptimalDistributionMultiItemBaselineAgent,
)
from environments import (
    Environment,
    NonStochasticAbruptChangeEnvironment,
    NonStochasticSmoothChangeEnvironment,
    StochasticEnvironment,
)
from plotting import (
    plot_animated_price_frequency_histograms,
    plot_budget_evolution,
    plot_conversion_rates,
    plot_cumulative_regret,
    plot_price_frequency_histograms,
)

# ---- Core simulation helpers (from req1.py) ----


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


def run_simulation(
    env: Environment, agent: Agent, prices: NDArray[np.float64]
) -> RunSimulationResult:
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
            env.num_items,
        ), f"Expected shape {(env.num_items,)}, got {price_indexes.shape}"

        # Environment interaction
        valuations = env.round(t)
        assert valuations.shape == (
            env.num_items,
        ), f"Expected shape {(env.num_items,)}, got {valuations.shape}"

        # Determine purchases and rewards
        rewards = np.zeros(num_items, dtype=np.float64)
        is_budget_exhausted = False
        for i in range(num_items):
            if price_indexes[i] == -1:
                assert np.all(
                    price_indexes == -1
                ), f"Once an item has budget exhausted (-1), all subsequent items must also be -1. Instead got {price_indexes}"
                is_budget_exhausted = True
                break  # Budget exhausted for this item, from now on -1 by default

            price = prices[price_indexes[i]]
            if valuations[i] >= price:
                rewards[i] = price

        if is_budget_exhausted:
            break  # Stop simulation if any item's budget is exhausted

        # Update agent with rewards
        agent.update(rewards, valuations=valuations)

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


AgentConfigType = TypeVar("AgentConfigType")
BaselineConfigType = TypeVar("BaselineConfigType")


def run_multiple_simulations(
    env_builder: Callable[[], Environment],
    agent_builder: Callable[[AgentConfigType], Agent],
    baseline_builder: Callable[[BaselineConfigType, Environment], Agent],
    num_trials: int,
    agent_config: AgentConfigType,
    baseline_config: BaselineConfigType,
    prices: NDArray[np.float64],
) -> RunMultipleSimulationsResult:
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

    valuations = np.zeros((num_trials, num_items, time_horizon), dtype=np.float64)

    agent_played_arms = np.full(
        (num_trials, num_items, time_horizon), -1, dtype=np.int64
    )
    baseline_played_arms = np.full(
        (num_trials, num_items, time_horizon), -1, dtype=np.int64
    )

    for trial in range(num_trials):
        print(f"Running trial {trial + 1}/{num_trials}...")
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
        baseline_played_arms=baseline_played_arms,
    )


# ---- Task: Multiple items (3) with budget constraint ----

print("Task: Multiple items (3) with budget constraint")

num_trials = 2
time_horizon = 12_000
prices = np.linspace(0.1, 1.0, 10)
num_prices = len(prices)
num_items = 3
budget = 12_000


def env_builder1() -> Environment:
    low = NonStochasticAbruptChangeEnvironment.gaussian_distribution(
        mean=0.25, std=0.05
    )
    medium = NonStochasticAbruptChangeEnvironment.gaussian_distribution(
        mean=0.5, std=0.05
    )
    high = NonStochasticAbruptChangeEnvironment.gaussian_distribution(
        mean=0.75, std=0.05
    )
    return NonStochasticAbruptChangeEnvironment(
        distribution_functions=[
            [low, high, low],
            [high, low, high],
            [medium, medium, medium],
        ],
        num_rounds=time_horizon,
    )


def env_builder2() -> Environment:
    low = NonStochasticSmoothChangeEnvironment.generate_beta_valuations(
        time_horizon=time_horizon, freq=1
    )
    medium = NonStochasticSmoothChangeEnvironment.generate_beta_valuations(
        time_horizon=time_horizon, freq=2
    )
    high = NonStochasticSmoothChangeEnvironment.generate_beta_valuations(
        time_horizon=time_horizon, freq=3
    )
    return NonStochasticSmoothChangeEnvironment(
        distribution_functions=[low, medium, high],
        num_rounds=time_horizon,
    )


@dataclass
class CombinatorialUCBBiddingSlidingWindowConfig:
    num_items: int
    num_prices: int
    budget: int
    alpha: float = 1.0


def combinatorial_agent_builder(
    config: CombinatorialUCBBiddingSlidingWindowConfig,
) -> Agent:
    assert isinstance(
        config, CombinatorialUCBBiddingSlidingWindowConfig
    ), f"Expected CombinatorialUCBBiddingSlidingWindowConfig, got {type(config)}"
    return CombinatorialUCBBiddingSlidingWindow(
        num_items=config.num_items,
        price_set=prices,
        budget=config.budget,
        time_horizon=time_horizon,
        window_size=500,
    )


@dataclass
class BaselineAgentConfig:
    # if None, no constraint in scheduling (evaluation still considers budget)
    budget: int | None = None


def baseline_builder(config: BaselineAgentConfig, env: Environment) -> Agent:
    assert isinstance(
        config, BaselineAgentConfig
    ), f"Expected BaselineAgentConfig, got {type(config)}"
    return FixedActionBaselineAgent(
        prices=prices,
        num_items=env.num_items,
        valuations=env.valuations,
        time_horizon=time_horizon,
        budget=config.budget,
    )


results = run_multiple_simulations(
    env_builder=env_builder2,
    agent_builder=combinatorial_agent_builder,
    baseline_builder=baseline_builder,
    num_trials=num_trials,
    agent_config=CombinatorialUCBBiddingSlidingWindowConfig(
        num_items=num_items, num_prices=num_prices, budget=budget
    ),
    baseline_config=BaselineAgentConfig(budget=budget),
    prices=prices,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_cumulative_regret(
    valuations=results.valuations,
    agents_played_arms=results.agent_played_arms[np.newaxis, ...],
    baseline_played_arms=results.baseline_played_arms,
    prices=prices,
    agents_names=["Sliding Window UCB"],
    title="Cumulative Regret: Sliding Window UCB vs Optimal Baseline",
    ax=axes[0],
)

plot_budget_evolution(
    valuations=results.valuations,
    agents_played_arms=results.agent_played_arms[np.newaxis, ...],
    prices=prices,
    agents_names=["Sliding Window UCB"],
    initial_budget=budget,
    ax=axes[1],
)

fig.savefig("req5_cumulative_regret_budget_evolution.png")


temp_env = env_builder2()
plot_price_frequency_histograms(
    valuations=results.valuations,
    agents_played_arms=results.baseline_played_arms[np.newaxis, :, :, :],
    prices=prices,
    agents_names=[f"Optimal Baseline"],
    save_plot=True,
    save_path_prefix=f"req5_price_histogram_baseline",
)

for interval_index in range(3):
    interval_length = time_horizon // 3
    start_time = interval_index * interval_length
    end_time = start_time + interval_length

    plot_conversion_rates(
        valuations=results.valuations[:, :, start_time:end_time],
        agents_played_arms=results.agent_played_arms[
            np.newaxis, :, :, start_time:end_time
        ],
        baseline_played_arms=results.baseline_played_arms[:, :, start_time:end_time],
        prices=prices,
        agents_names=[f"Sliding Window UCB (Interval {interval_index + 1})"],
        save_plot=True,
        save_path=f"req5_conversion_rates_interval_{interval_index + 1}.png",
    )

    plot_price_frequency_histograms(
        valuations=results.valuations[:, :, start_time:end_time],
        agents_played_arms=results.agent_played_arms[
            np.newaxis, :, :, start_time:end_time
        ],
        prices=prices,
        agents_names=[f"Sliding Window UCB (Interval {interval_index + 1})"],
        save_plot=True,
        save_path_prefix=f"req5_price_histogram_interval",
    )

# Genera e salva animazione per l'agente Sliding Window UCB
# print("Generando animazione per l'agente Sliding Window UCB...")
# plot_animated_price_frequency_histograms(
#     valuations=results.valuations,
#     agents_played_arms=results.agent_played_arms[np.newaxis, ...],
#     prices=prices,
#     number_of_frames=100,
#     agents_names=["Sliding Window UCB"],
#     save_path_prefix="req5_animation_sliding_window_ucb",
# )

plt.show()

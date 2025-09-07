from dataclasses import dataclass
from typing import Callable, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from agents import (
    Agent,
    BanditFeedbackPrimalDual,
    FFPrimalDualPricingAgent,
    PrimalDualAgent,
)
from baselines import (
    FixedActionBaselineAgent,
    OptimalDistributionSingleItemBaselineAgent,
)
from environments import (
    Environment,
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
        agent.update(rewards, full_rewards=valuations)

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

    valuations = np.zeros(
        (num_trials, num_items, time_horizon), dtype=np.float64)

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
time_horizon = 20_000
prices = np.linspace(0.1, 1.0, 10)
num_prices = len(prices)
num_items = 1
budget = 8_000


def env_builder() -> Environment:
    return NonStochasticSmoothChangeEnvironment(
        distribution_functions=[
            NonStochasticSmoothChangeEnvironment.gaussian_distribution(
                mean=0.25, std=0.1, freq=10, magnitude=0.4
            ),
        ],
        num_rounds=time_horizon,
    )


@dataclass
class FullFeedbackPrimalDualConfig:
    num_items: int
    num_prices: int
    budget: int
    alpha: float = 1.0


def combinatorial_agent_builder(config: FullFeedbackPrimalDualConfig) -> Agent:
    assert isinstance(
        config, FullFeedbackPrimalDualConfig
    ), f"Expected PrimalDualAgentConfig, got {type(config)}"
    return FFPrimalDualPricingAgent(
        prices=prices,
        B=config.budget,
        T=time_horizon,
    )


@dataclass
class BaselineAgentConfig:
    # if None, no constraint in scheduling (evaluation still considers budget)
    budget: int | None = None


def baseline_builder(config: BaselineAgentConfig, env: Environment) -> Agent:
    assert isinstance(
        config, BaselineAgentConfig
    ), f"Expected BaselineAgentConfig, got {type(config)}"
    # return OptimalDistributionSingleItemBaselineAgent(
    #     prices=prices,
    #     valuations=env.valuations,
    #     time_horizon=time_horizon,
    #     budget=config.budget,
    # )
    return FixedActionBaselineAgent(
        prices=prices,
        num_items=1,
        time_horizon=time_horizon,
        valuations=env.valuations,
        budget=budget
    )


results = run_multiple_simulations(
    env_builder=env_builder,
    agent_builder=combinatorial_agent_builder,
    baseline_builder=baseline_builder,
    num_trials=num_trials,
    agent_config=FullFeedbackPrimalDualConfig(
        num_items=num_items, num_prices=num_prices, budget=budget, alpha=0.01
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
    agents_names=["Primal Dual"],
    title="Cumulative Regret: Primal Dual vs Baseline",
    ax=axes[0],
)

plot_budget_evolution(
    valuations=results.valuations,
    agents_played_arms=results.agent_played_arms[np.newaxis, ...],
    prices=prices,
    agents_names=["Primal Dual"],
    initial_budget=budget,
    ax=axes[1],
)

fig.savefig("req3_cumulative_regret_and_budget_evolution.png")

# Conversion rates as a separate plot with dual subplots
plot_conversion_rates(
    valuations=results.valuations,
    agents_played_arms=results.agent_played_arms[np.newaxis, ...],
    baseline_played_arms=results.baseline_played_arms,
    prices=prices,
    agents_names=["Primal Dual"],
    save_path="req3_conversion_rates.png",
    save_plot=True
)

plot_price_frequency_histograms(
    valuations=results.valuations,
    agents_played_arms=results.agent_played_arms[np.newaxis, ...],
    prices=prices,
    agents_names=["Primal Dual"],
    save_plot=True,
    save_path_prefix="req3_price_frequency"
)

plt.show()

# Genera e salva animazione per l'agente Primal Dual
print("Generando animazione per l'agente Primal Dual...")
plot_animated_price_frequency_histograms(
    valuations=results.valuations,
    agents_played_arms=results.agent_played_arms[np.newaxis, ...],
    prices=prices,
    agents_names=["Primal Dual"],
    save_path_prefix="req3_animation_primal_dual"
)

plt.show()

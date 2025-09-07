from dataclasses import dataclass
from typing import Callable
from numpy.typing import NDArray
import numpy as np
from matplotlib import pyplot as plt

from environments import Environment, NonStochasticSmoothChangeEnvironment
from agents import Agent, MultiItemDualPricingAgent, CombinatorialUCBBidding, CombinatorialUCBBiddingSlidingWindow
from baselines import FixedActionBaselineAgent
from plotting import (
    plot_price_frequency_histograms,
    plot_cumulative_regret,
    plot_budget_evolution,
    plot_animated_price_frequency_histograms,
    plot_conversion_rates,
)


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
        played_arms: Played arms tensor (num_agents, num_trials, num_items, time_horizon) (-1 if not played else arm_index)
    """

    valuations: NDArray[np.float64]  # (num_trials, num_items, time_horizon)
    # (num_trials, num_items, time_horizon) (-1/int)
    agents_played_arms: NDArray[np.int64]
    baseline_played_arms: NDArray[np.int64]


def run_multiple_simulations(
    env_builder: Callable[[], Environment],
    agent_builders: list[Callable[[Environment], Agent]],
    baseline_builder: Callable[[Environment], Agent],
    num_trials: int,
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
    num_agents = len(agent_builders)

    valuations = np.zeros(
        (num_trials, num_items, time_horizon), dtype=np.float64)

    agents_played_arms = np.full(
        (num_agents, num_trials, num_items, time_horizon), -1, dtype=np.int64
    )
    baseline_played_arms = np.full(
        (num_trials, num_items, time_horizon), -1, dtype=np.int64
    )

    for trial in range(num_trials):
        print(f"Running trial {trial + 1}/{num_trials}...")
        env = env_builder()

        baseline_agent = baseline_builder(env)
        baseline_results = run_simulation(env, baseline_agent, prices)
        baseline_played_arms[trial] = baseline_results.played_arms

        for i, agent_builder in enumerate(agent_builders):
            print(f"  Running agent {i + 1}/{num_agents}...")
            agent = agent_builder(env)
            agent_result = run_simulation(env, agent, prices)
            agents_played_arms[i, trial] = agent_result.played_arms

        valuations[trial] = env.valuations

    return RunMultipleSimulationsResult(
        valuations=valuations,
        agents_played_arms=agents_played_arms,
        baseline_played_arms=baseline_played_arms,
    )


# ---- Task: Multiple items (3) with budget constraint ----

print("Task: Multiple items (3) with budget constraint")

num_trials = 2
time_horizon = 10_000
prices = np.linspace(0.1, 1.0, 10)
num_prices = len(prices)
num_items = 3
budget = 8_000

primal_dual_eta = 1 / np.sqrt(time_horizon)


def env_builder() -> Environment:
    return NonStochasticSmoothChangeEnvironment(
        distribution_functions=[
            NonStochasticSmoothChangeEnvironment.generate_simple_tv(
                time_horizon, 1),
            NonStochasticSmoothChangeEnvironment.generate_beta_valuations(
                time_horizon, 50
            ),
            NonStochasticSmoothChangeEnvironment.gaussian_distribution(
                mean=0.55, std=0.1, freq=1),
        ],
        num_rounds=time_horizon,
    )


def combinatorial_agent_builder(env: Environment) -> Agent:
    return CombinatorialUCBBidding(
        num_items=env.num_items,
        price_set=prices,
        budget=budget,
        time_horizon=env.time_horizon,
    )


def primal_dual_agent_builder(env: Environment) -> Agent:
    return MultiItemDualPricingAgent(
        prices=prices,
        B=budget,
        T=time_horizon,
        n_products=num_items,
        eta=primal_dual_eta,
    )


def sliding_window_agent_builder(env: Environment) -> Agent:
    return CombinatorialUCBBiddingSlidingWindow(
        num_items=num_items,
        price_set=prices,
        budget=budget,
        time_horizon=env.time_horizon,
        window_size=150
    )


def baseline_builder(env: Environment) -> Agent:
    return FixedActionBaselineAgent(
        num_items=num_items,
        prices=prices,
        time_horizon=time_horizon,
        valuations=env.valuations,
        budget=budget
    )


results = run_multiple_simulations(
    env_builder=env_builder,
    agent_builders=[
        primal_dual_agent_builder,
        combinatorial_agent_builder,
        sliding_window_agent_builder,
    ],
    baseline_builder=baseline_builder,
    num_trials=num_trials,
    prices=prices,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_cumulative_regret(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[0][np.newaxis, ...],
    baseline_played_arms=results.baseline_played_arms,
    prices=prices,
    agents_names=["Primal Dual"],
    title="Cumulative Regret: MultiProduct Primal Dual",
    ax=axes[0],
)

plot_budget_evolution(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[0][np.newaxis, ...],
    prices=prices,
    agents_names=["Primal Dual"],
    initial_budget=budget,
    ax=axes[1],
)

fig.savefig("req4_primaldual_cumregret_budgetevolution.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_cumulative_regret(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[0, 1], ...],
    baseline_played_arms=results.baseline_played_arms,
    prices=prices,
    agents_names=["Primal Dual", "Combinatorial UCB"],
    title="Cumulative Regret: MultiProduct Primal Dual vs Combinatorial UCB",
    ax=axes[0],
)

plot_budget_evolution(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[0, 1], ...],
    prices=prices,
    agents_names=["Primal Dual", "Combinatorial UCB"],
    initial_budget=budget,
    ax=axes[1],
)

fig.savefig("req4_primaldual_vs_combucb_cumregret_budgetevolution.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_cumulative_regret(
    valuations=results.valuations,
    # Pass only agent 0 and 2
    agents_played_arms=results.agents_played_arms[[0, 2], ...],
    baseline_played_arms=results.baseline_played_arms,
    prices=prices,
    agents_names=["Primal Dual", "UCB Sliding Window"],
    title="Cumulative Regret: MultiProduct Primal Dual vs UCB Sliding Window",
    ax=axes[0],
)

plot_budget_evolution(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[0, 2], ...],
    prices=prices,
    agents_names=["Primal Dual", "UCB Sliding Window"],
    initial_budget=budget,
    ax=axes[1],
)

fig.savefig("req4_primaldual_vs_ucbsliding_cumregret_budgetevolution.png")

# Conversion rates as a separate plot with dual subplots
plot_conversion_rates(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[0][np.newaxis, ...],
    baseline_played_arms=results.baseline_played_arms,
    prices=prices,
    agents_names=["Primal Dual"],
    save_plot=True,
    save_path="req4_primaldual_conversion_rates.png",
)

plot_price_frequency_histograms(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms,
    prices=prices,
    agents_names=["Primal Dual", "Combinatorial UCB", "UCB Sliding Window"],
    save_plot=True,
    save_path_prefix="req4_primaldual_price_histograms"
)

plt.show()

# Genera e salva animazione per l'agente MultiProduct Primal Dual
print("Generando animazione per l'agente MultiProduct Primal Dual...")
plot_animated_price_frequency_histograms(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[0][np.newaxis, ...],
    prices=prices,
    agents_names=["Primal Dual"],
    save_path_prefix="req4_animation_primal_dual"
)

plt.show()

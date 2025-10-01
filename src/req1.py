from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from core.environments import Environment, StochasticEnvironment
from core.agents import Agent, UCBAgent, CombinatorialUCBBidding
from core.baselines import FixedActionBaselineAgent
from core.plotting import plot_price_frequency_histograms, plot_cumulative_regret, plot_budget_evolution, plot_animated_price_frequency_histograms, plot_conversion_rates
from core.runner import run_multiple_simulations

results_dir = Path("../results/req1")
results_dir.mkdir(parents=True, exist_ok=True)

print("Task 1.1: Without budget constraint")

num_trials = 2
time_horizon = 10000
num_prices = 10
prices = np.linspace(0.1, 1, num_prices, dtype=np.float64)


def env_builder() -> Environment:
    return StochasticEnvironment(
        distribution_functions=[
            StochasticEnvironment.gaussian_distribution(mean=0.5, std=0.1)],
        num_rounds=time_horizon,
    )


def agent_builder(env: Environment) -> Agent:
    return UCBAgent(
        num_prices=num_prices,
    )


def baseline_builder(env: Environment) -> Agent:
    return FixedActionBaselineAgent(
        num_items=env.num_items,
        prices=prices,
        time_horizon=time_horizon,
        valuations=env.valuations,
    )


results = run_multiple_simulations(
    env_builder=env_builder,
    agent_builders=[
        baseline_builder,
        agent_builder,
    ],
    num_trials=num_trials,
    prices=prices
)

plot_cumulative_regret(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[1], ...],
    baseline_played_arms=results.agents_played_arms[0],
    prices=prices,
    agents_names=["UCB Agent"],
    title="Cumulative Regret of UCB Agent vs Random Baseline",
    save_plot=True,
    save_path=results_dir / "task1_1_cumulative_regret.png",
)

plot_price_frequency_histograms(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[1], ...],
    prices=prices,
    agents_names=["UCB Agent"],
    save_plot=True,
    save_path_prefix=results_dir / "task1_1_ucb_agent_histogram"
)

print("Task 1.1.2:  Without budget constraint but more arms")

# num_trials = 2
# time_horizon = 10000
num_prices = 100
prices = np.linspace(0.1, 1, num_prices, dtype=np.float64)


def env_builder() -> Environment:
    return StochasticEnvironment(
        distribution_functions=[
            StochasticEnvironment.gaussian_distribution(mean=0.5, std=0.1)],
        num_rounds=time_horizon,
    )


def agent_builder(env: Environment) -> Agent:
    return UCBAgent(
        num_prices=num_prices,
    )


def baseline_builder(env: Environment) -> Agent:
    return FixedActionBaselineAgent(
        num_items=env.num_items,
        prices=prices,
        time_horizon=time_horizon,
        valuations=env.valuations,
    )


results = run_multiple_simulations(
    env_builder=env_builder,
    agent_builders=[
        baseline_builder,
        agent_builder,
    ],
    num_trials=num_trials,
    prices=prices
)

plot_cumulative_regret(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[1], ...],
    baseline_played_arms=results.agents_played_arms[0],
    prices=prices,
    agents_names=["UCB Agent"],
    title="Cumulative Regret of UCB Agent vs Random Baseline",
    save_plot=True,
    save_path=results_dir / "task1_1_2_cumulative_regret.png",
)

plot_price_frequency_histograms(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[1], ...],
    prices=prices,
    agents_names=["UCB Agent"],
    save_plot=True,
    save_path_prefix=results_dir / "task1_1_2_ucb_agent_histogram"
)

print("Task 1.2: With budget constraint")

# Same env builder as before

time_horizon = 10000
budget = 3000
num_prices = 10
prices = np.linspace(0.1, 1, num_prices, dtype=np.float64)


def combinatorial_agent_builder(env: Environment) -> Agent:
    return CombinatorialUCBBidding(
        num_items=1,  # Currently only supports single item
        price_set=prices,
        budget=budget,
        time_horizon=time_horizon,
    )


results = run_multiple_simulations(
    env_builder=env_builder,
    agent_builders=[
        baseline_builder,
        combinatorial_agent_builder,
    ],
    num_trials=num_trials,
    prices=prices,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_cumulative_regret(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[1], ...],
    baseline_played_arms=results.agents_played_arms[0],
    prices=prices,
    agents_names=["UCB Agent"],
    title="Cumulative Regret of UCB Agent vs Random Baseline",
    ax=axes[0]
)

plot_budget_evolution(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[1], ...],
    prices=prices,
    initial_budget=budget,
    agents_names=["UCB Agent"],
    ax=axes[1]
)

fig.savefig(results_dir / "task1_2_cumulative_regret_and_budget_evolution.png")

# Conversion rates as a separate plot with dual subplots
plot_conversion_rates(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[1], ...],
    baseline_played_arms=results.agents_played_arms[0],
    prices=prices,
    agents_names=["UCB Agent"],
    save_plot=True,
    save_path=results_dir / "task1_2_conversion_rates.png"
)

plot_price_frequency_histograms(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[1], ...],
    prices=prices,
    agents_names=["UCB Agent"],
    save_plot=True,
    save_path_prefix=results_dir / "task1_2_ucb_agent_histogram"
)

# Genera e salva animazione per l'agente UCB
print("Generando animazione per l'agente UCB...")
plot_animated_price_frequency_histograms(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[1], ...],
    prices=prices,
    agents_names=["UCB Agent"],
    save_path_prefix=results_dir / "animation_budget_ucb"
)

plt.show()

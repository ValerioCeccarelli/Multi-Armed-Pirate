from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from core.environments import Environment, StochasticEnvironment
from core.agents import Agent, CombinatorialUCBBidding
from core.baselines import FixedActionBaselineAgent
from core.plotting import (
    plot_price_frequency_histograms,
    plot_cumulative_regret,
    plot_budget_evolution,
    plot_animated_price_frequency_histograms,
    plot_conversion_rates,
)
from core.runner import run_multiple_simulations

results_dir = Path("../results/req2")
results_dir.mkdir(parents=True, exist_ok=True)

# ---- Task: Multiple items (3) with budget constraint ----

print("Task: Multiple items (3) with budget constraint")

num_trials = 2
time_horizon = 20_000
num_prices = 10
prices = np.linspace(0.1, 1.0, num_prices, dtype=np.float64)
num_items = 3
budget = 18000  # integer budget


def env_builder() -> Environment:
    return StochasticEnvironment(
        distribution_functions=[
            StochasticEnvironment.gaussian_distribution(mean=0.30, std=0.1),
            StochasticEnvironment.gaussian_distribution(mean=0.45, std=0.1),
            StochasticEnvironment.gaussian_distribution(mean=0.55, std=0.1),
        ],
        num_rounds=time_horizon,
    )


def combinatorial_agent_builder(env: Environment) -> Agent:
    return CombinatorialUCBBidding(
        num_items=env.num_items,
        price_set=prices,
        budget=budget,
        time_horizon=time_horizon,
    )


def baseline_builder(env: Environment) -> Agent:
    return FixedActionBaselineAgent(
        num_items=env.num_items,
        prices=prices,
        time_horizon=time_horizon,
        valuations=env.valuations,
        budget=budget,
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
    agents_names=["Combinatorial UCB"],
    title="Cumulative Regret: Combinatorial UCB vs Fixed Baseline",
    ax=axes[0],
)

plot_budget_evolution(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[1], ...],
    prices=prices,
    agents_names=["Combinatorial UCB"],
    initial_budget=budget,
    ax=axes[1],
)

fig.savefig(results_dir / "task2_cumulative_regret_and_budget_evolution.png")

# Conversion rates as a separate plot with dual subplots
plot_conversion_rates(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[1], ...],
    baseline_played_arms=results.agents_played_arms[0],
    prices=prices,
    agents_names=["Combinatorial UCB"],
    save_plot=True,
    save_path=results_dir / "task2_conversion_rates.png"
)

plot_price_frequency_histograms(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[1], ...],
    prices=prices,
    agents_names=["Combinatorial UCB"],
    save_plot=True,
    save_path_prefix=results_dir / "task2_price_frequency_histograms.png"
)

# Genera e salva animazione per l'agente Combinatorial UCB
print("Generando animazione per l'agente Combinatorial UCB...")
plot_animated_price_frequency_histograms(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[1], ...],
    prices=prices,
    agents_names=["Combinatorial UCB"],
    save_path_prefix=results_dir / "animation_combinatorial_ucb"
)

plt.show()

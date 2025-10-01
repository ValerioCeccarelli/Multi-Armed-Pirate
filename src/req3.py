import os
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from agents import (
    Agent,
    PrimalDualAgent,
)
from baselines import FixedActionBaselineAgent
from environments import (
    Environment,
    NonStochasticSmoothChangeEnvironment,
)
from plotting import (
    plot_animated_price_frequency_histograms,
    plot_budget_evolution,
    plot_conversion_rates,
    plot_cumulative_regret,
    plot_price_frequency_histograms,
)
from runner import run_multiple_simulations

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


def primal_dual_agent_builder(env: Environment) -> Agent:
    return PrimalDualAgent(
        prices=prices,
        budget=budget,
        time_horizon=time_horizon,
        eta=0.01,
    )


def baseline_builder(env: Environment) -> Agent:
    return FixedActionBaselineAgent(
        prices=prices,
        num_items=1,
        time_horizon=time_horizon,
        valuations=env.valuations,
        budget=budget
    )


results = run_multiple_simulations(
    env_builder=env_builder,
    agent_builders=[
        baseline_builder,
        primal_dual_agent_builder,
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
    agents_names=["Primal Dual"],
    title="Cumulative Regret: Primal Dual vs Baseline",
    ax=axes[0],
)

plot_budget_evolution(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[1], ...],
    prices=prices,
    agents_names=["Primal Dual"],
    initial_budget=budget,
    ax=axes[1],
)

os.makedirs("req3", exist_ok=True)
fig.savefig("req3/cumulative_regret_and_budget_evolution.png")

# Conversion rates as a separate plot with dual subplots
plot_conversion_rates(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[1], ...],
    baseline_played_arms=results.agents_played_arms[0],
    prices=prices,
    agents_names=["Primal Dual"],
    save_path="req3/conversion_rates.png",
    save_plot=True
)

plot_price_frequency_histograms(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[1], ...],
    prices=prices,
    agents_names=["Primal Dual"],
    save_plot=True,
    save_path_prefix="req3/price_frequency"
)

plt.show()

# Genera e salva animazione per l'agente Primal Dual
print("Generando animazione per l'agente Primal Dual...")
plot_animated_price_frequency_histograms(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[1], ...],
    prices=prices,
    agents_names=["Primal Dual"],
    save_path_prefix="req3/animation_primal_dual"
)

plt.show()

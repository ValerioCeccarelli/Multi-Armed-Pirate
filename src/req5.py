from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from core.agents import Agent, CombinatorialUCBBiddingSlidingWindow
from core.baselines import FixedActionBaselineAgent
from core.environments import (
    Environment,
    NonStochasticAbruptChangeEnvironment,
    NonStochasticSmoothChangeEnvironment,
)
from core.plotting import (
    plot_animated_price_frequency_histograms,
    plot_budget_evolution,
    plot_conversion_rates,
    plot_cumulative_regret,
    plot_price_frequency_histograms,
)
from core.runner import run_multiple_simulations

results_dir = Path("../results/req5")
results_dir.mkdir(parents=True, exist_ok=True)

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


def combinatorial_agent_builder(env: Environment) -> Agent:
    return CombinatorialUCBBiddingSlidingWindow(
        num_items=env.num_items,
        price_set=prices,
        budget=budget,
        time_horizon=env.time_horizon,
        window_size=500,
    )


def baseline_builder(env: Environment) -> Agent:
    return FixedActionBaselineAgent(
        prices=prices,
        num_items=env.num_items,
        valuations=env.valuations,
        time_horizon=time_horizon,
        budget=budget,
    )


results = run_multiple_simulations(
    env_builder=env_builder2,
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
    agents_names=["Sliding Window UCB"],
    title="Cumulative Regret: Sliding Window UCB vs Optimal Baseline",
    ax=axes[0],
)

plot_budget_evolution(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[1], ...],
    prices=prices,
    agents_names=["Sliding Window UCB"],
    initial_budget=budget,
    ax=axes[1],
)

fig.savefig(results_dir / "cumulative_regret_budget_evolution.png")


temp_env = env_builder2()
plot_price_frequency_histograms(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[0], ...],
    prices=prices,
    agents_names=[f"Optimal Baseline"],
    save_plot=True,
    save_path_prefix=results_dir / "price_histogram_baseline",
)

for interval_index in range(3):
    interval_length = time_horizon // 3
    start_time = interval_index * interval_length
    end_time = start_time + interval_length

    plot_conversion_rates(
        valuations=results.valuations[:, :, start_time:end_time],
        agents_played_arms=results.agents_played_arms[
            [1], :, :, start_time:end_time
        ],
        baseline_played_arms=results.agents_played_arms[0,
                                                        :, :, start_time:end_time],
        prices=prices,
        agents_names=[f"Sliding Window UCB (Interval {interval_index + 1})"],
        save_plot=True,
        save_path=results_dir / "conversion_rates_interval_{interval_index + 1}.png",
    )

    plot_price_frequency_histograms(
        valuations=results.valuations[:, :, start_time:end_time],
        agents_played_arms=results.agents_played_arms[
            [1], :, :, start_time:end_time
        ],
        prices=prices,
        agents_names=[f"Sliding Window UCB (Interval {interval_index + 1})"],
        save_plot=True,
        save_path_prefix=results_dir / "price_histogram_interval",
    )

# Genera e salva animazione per l'agente Sliding Window UCB
# print("Generando animazione per l'agente Sliding Window UCB...")
plot_animated_price_frequency_histograms(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[[1], ...],
    prices=prices,
    agents_names=["Sliding Window UCB"],
    save_path_prefix=results_dir / "animation_sliding_window_ucb",
)

plt.show()

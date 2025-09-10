import numpy as np
from matplotlib import pyplot as plt

from environments import Environment, NonStochasticSmoothChangeEnvironment
from agents import Agent, MultiItemPrimalDualAgent, CombinatorialUCBBidding, CombinatorialUCBBiddingSlidingWindow
from baselines import FixedActionBaselineAgent
from plotting import (
    plot_price_frequency_histograms,
    plot_cumulative_regret,
    plot_budget_evolution,
    plot_animated_price_frequency_histograms,
    plot_conversion_rates,
)
from runner import run_multiple_simulations

# ---- Task: Multiple items (3) with budget constraint ----

print("Task: Multiple items (3) with budget constraint")

num_trials = 2
time_horizon = 1_000
prices = np.linspace(0.1, 1.0, 10)
num_prices = len(prices)
num_items = 3
budget = 800

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
    return MultiItemPrimalDualAgent(
        prices=prices,
        budget=budget,
        time_horizon=time_horizon,
        n_products=num_items,
        eta=primal_dual_eta,
    )


def prima_dul_dual_agent_with_dynamic_rho_builder(env: Environment) -> Agent:
    return MultiItemPrimalDualAgent(
        prices=prices,
        budget=budget,
        time_horizon=time_horizon,
        n_products=num_items,
        eta=primal_dual_eta,
        dynamic_rho=True
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
        baseline_builder,
        primal_dual_agent_builder,
        combinatorial_agent_builder,
        sliding_window_agent_builder,
        prima_dul_dual_agent_with_dynamic_rho_builder,
    ],
    num_trials=num_trials,
    prices=prices,
)

baseline_played_arms = results.agents_played_arms[0]
only_primal_dual_mask = [1]
primal_dual_vs_combucb_mask = [1, 2]
primal_dual_vs_ucbsliding_mask = [1, 3]
static_vs_dynamic_rho_mask = [1, 4]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_cumulative_regret(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[only_primal_dual_mask, ...],
    baseline_played_arms=baseline_played_arms,
    prices=prices,
    agents_names=["Primal Dual"],
    title="Cumulative Regret: MultiProduct Primal Dual",
    ax=axes[0],
)

plot_budget_evolution(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[only_primal_dual_mask, ...],
    prices=prices,
    agents_names=["Primal Dual"],
    initial_budget=budget,
    ax=axes[1],
)

fig.savefig("req4/primaldual_cumregret_budgetevolution.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_cumulative_regret(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[primal_dual_vs_combucb_mask, ...],
    baseline_played_arms=baseline_played_arms,
    prices=prices,
    agents_names=["Primal Dual", "Combinatorial UCB"],
    title="Cumulative Regret: MultiProduct Primal Dual vs Combinatorial UCB",
    ax=axes[0],
)

plot_budget_evolution(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[primal_dual_vs_combucb_mask, ...],
    prices=prices,
    agents_names=["Primal Dual", "Combinatorial UCB"],
    initial_budget=budget,
    ax=axes[1],
)

fig.savefig("req4/primaldual_vs_combucb_cumregret_budgetevolution.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_cumulative_regret(
    valuations=results.valuations,
    # Pass only agent 0 and 2
    agents_played_arms=results.agents_played_arms[primal_dual_vs_ucbsliding_mask, ...],
    baseline_played_arms=baseline_played_arms,
    prices=prices,
    agents_names=["Primal Dual", "UCB Sliding Window"],
    title="Cumulative Regret: MultiProduct Primal Dual vs UCB Sliding Window",
    ax=axes[0],
)

plot_budget_evolution(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[primal_dual_vs_ucbsliding_mask, ...],
    prices=prices,
    agents_names=["Primal Dual", "UCB Sliding Window"],
    initial_budget=budget,
    ax=axes[1],
)

fig.savefig("req4/primaldual_vs_ucbsliding_cumregret_budgetevolution.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_cumulative_regret(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[static_vs_dynamic_rho_mask, ...],
    baseline_played_arms=baseline_played_arms,
    prices=prices,
    agents_names=["Fixed", "Dynamic"],
    title="Cumulative Regret: Fixed vs Dynamic Rho",
    ax=axes[0],
)

plot_budget_evolution(
    valuations=results.valuations,
    agents_played_arms=results.agents_played_arms[static_vs_dynamic_rho_mask, ...],
    prices=prices,
    agents_names=["Fixed", "Dynamic"],
    initial_budget=budget,
    ax=axes[1],
)

fig.savefig("req4/primaldual_fixed_vs_dynamic_cumregret_budgetevolution.png")

# Conversion rates as a separate plot with dual subplots
# plot_conversion_rates(
#     valuations=results.valuations,
#     agents_played_arms=results.agents_played_arms[[1], ...],
#     baseline_played_arms=results.baseline_played_arms,
#     prices=prices,
#     agents_names=["Primal Dual"],
#     save_plot=True,
#     save_path="req4/primaldual_conversion_rates.png",
# )

# plot_price_frequency_histograms(
#     valuations=results.valuations,
#     agents_played_arms=results.agents_played_arms[[1,2,3],...],
#     prices=prices,
#     agents_names=["Primal Dual", "Combinatorial UCB", "UCB Sliding Window"],
#     save_plot=True,
#     save_path_prefix="req4/primaldual_price_histograms"
# )

plt.show()

# Genera e salva animazione per l'agente MultiProduct Primal Dual
# print("Generando animazione per l'agente MultiProduct Primal Dual...")
# plot_animated_price_frequency_histograms(
#     valuations=results.valuations,
#     agents_played_arms=results.agents_played_arms[only_primal_dual_mask, ...],
#     prices=prices,
#     agents_names=["Primal Dual"],
#     save_path_prefix="req4/animation_primal_dual"
# )

# plt.show()

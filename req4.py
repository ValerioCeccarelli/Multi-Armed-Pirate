"""
Main script for executing combinatorial pricing simulation using the PrimalDualAgent.

This script mirrors the structure of req2.py but swaps in the EXP3-based
primal-dual pacing agent defined in src/agent.py (imported from req3cecc.py).
"""

import random
import numpy as np
from src.agent import PrimalDualAgent
from src.core import (
    simulate,
    ExperimentConfig,
    EnvironmentConfig,
    run_single_simulation,
)
from src.environment import Environment
from src.plotting import plot_statistics, PlottingConfig, create_animated_histogram


# --------------------- Utilities ---------------------

def setup_reproducible_environment(seed: int = 42) -> None:
    """Configure environment for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)


# --------------------- Config Builders ---------------------

# def create_experiment_config() -> ExperimentConfig:
#     """
#     Create complete configuration for the experiment (single item example).
#     """
#     env_config = EnvironmentConfig(
#         means=[25],
#         stds=[10],
#     )

#     experiment_config = ExperimentConfig(
#         time_horizon=1400,
#         budget=300,
#         price_range=list(range(5, 41, 5)),
#         num_trials=10,
#         exploration_param=0.01,  # used as beta for EXP3/dual by default
#         environment=env_config,
#     )

#     return experiment_config


def create_multi_item_experiment_config() -> ExperimentConfig:
    """
    Example configuration for a multi-item experiment (3 items).
    """
    env_config = EnvironmentConfig(
        means=[8, 10, 15],
        stds=[4, 5, 6],
    )

    experiment_config = ExperimentConfig(
        time_horizon=3000,
        budget=1000,
        price_range=list(range(8, 41, 2)),
        num_trials=10,
        exploration_param=4,
        environment=env_config,
    )
    return experiment_config


# --------------------- Builders ---------------------

def create_environment_builder(env_config: EnvironmentConfig, time_horizon: int):
    def builder():
        return Environment(
            mean=env_config.means,
            std=env_config.stds,
            time_horizon=time_horizon,
        )
    return builder


def create_agent_builder(experiment_config: ExperimentConfig):

    def builder():
        return PrimalDualAgent(
            price_set=experiment_config.price_range,
            num_items=experiment_config.num_items,
            B=experiment_config.budget,
            T=experiment_config.time_horizon,
            beta=experiment_config.exploration_param,
        )

    return builder


# --------------------- Helpers ---------------------

def print_experiment_summary(config: ExperimentConfig) -> None:
    print("EXPERIMENT PARAMETERS:")
    print(f"- Time horizon: {config.time_horizon} rounds")
    print(f"- Total budget: {config.budget} units")
    print(f"- Number of items: {config.num_items}")
    print(f"- Available prices: {config.price_range}")
    print(f"- Number of trials: {config.num_trials}")
    print(f"- Exploration parameter (beta): {config.exploration_param}")
    print(f"- Valuation means: {config.environment.means}")
    print(f"- Standard deviations: {config.environment.stds}")


# --------------------- Runners ---------------------

def run_experiment_with_animation() -> None:
    print("Starting primal-dual pricing simulation with animation...")
    print("=" * 60)

    setup_reproducible_environment()

    experiment_config = create_multi_item_experiment_config()

    print(f"Number of items: {experiment_config.num_items}")
    print_experiment_summary(experiment_config)
    print("=" * 60)

    env_builder = create_environment_builder(
        experiment_config.environment, experiment_config.time_horizon)
    agent_builder = create_agent_builder(experiment_config)

    # Single run with time series tracking for animation (via core.run_single_simulation)
    print("Running single simulation with time tracking for animation...")
    agent = agent_builder()
    env = env_builder()
    single_result = run_single_simulation(
        agent, env, experiment_config, track_time_series=True)

    # Animated histogram
    if single_result.time_series_data:
        print("Generating animated histogram...")
        baseline_prices = np.array(single_result.baseline_price_per_item)
        create_animated_histogram(
            single_result.time_series_data,
            baseline_prices,
            experiment_config.price_range,
            output_filename="agent_prices.gif",
            fps=2,
        )

    # Full multi-trial simulation
    print("Running full simulation...")
    result = simulate(
        agent_builder=agent_builder,
        env_builder=env_builder,
        config=experiment_config,
    )

    print("\nGenerating final plots...")
    plot_statistics(
        result,
        rounds_per_trial=experiment_config.time_horizon,
        n_trials=experiment_config.num_trials,
        initial_budget=experiment_config.budget,
    )

    print("Simulation with animation completed successfully!")


def run_experiment() -> None:
    print("Starting primal-dual pricing simulation...")
    print("=" * 60)

    setup_reproducible_environment()

    experiment_config = create_multi_item_experiment_config()

    print(f"Number of items: {experiment_config.num_items}")
    print_experiment_summary(experiment_config)
    print("=" * 60)

    env_builder = create_environment_builder(
        experiment_config.environment, experiment_config.time_horizon)
    agent_builder = create_agent_builder(experiment_config)

    print("Starting simulation...")
    result = simulate(
        agent_builder=agent_builder,
        env_builder=env_builder,
        config=experiment_config,
    )

    plotting_config = PlottingConfig(
        figure_size=(15, 10),
        dpi=150,
        save_plots=True,
        show_plots=True,
        output_filename="simulation_results.png",
    )

    print("\nGenerating comprehensive plots...")
    plot_statistics(
        result,
        rounds_per_trial=experiment_config.time_horizon,
        n_trials=experiment_config.num_trials,
        initial_budget=experiment_config.budget,
    )

    print("Simulation completed successfully!")


if __name__ == "__main__":
    run_experiment_with_animation()

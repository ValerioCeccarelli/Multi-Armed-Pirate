"""
Main script for executing combinatorial pricing simulation.

This script configures and runs a complete simulation of the combinatorial UCB
agent in a multi-item pricing environment with budget constraints.
"""

import random
import numpy as np

from src.agent import CombinatorialUCBBidding
from src.core import simulate, ExperimentConfig, EnvironmentConfig, run_single_simulation
from src.environment import Environment
from src.plotting import plot_statistics, PlottingConfig, create_animated_histogram


def setup_reproducible_environment(seed: int = 42) -> None:
    """
    Configure environment for reproducible results.
    
    Args:
        seed: Seed for random number generators
    """
    random.seed(seed)
    np.random.seed(seed)


def create_experiment_config() -> ExperimentConfig:
    """
    Create complete configuration for the experiment.
    
    Returns:
        Experiment configuration
    """
    # Environment configuration
    # Number of items is automatically inferred from the length of means and stds
    env_config = EnvironmentConfig(
        means=[25],  # Valuation means for each item
        stds=[10],    # Standard deviations for each item
    )
    
    # Complete experiment configuration
    # num_items is automatically inferred from env_config.num_items (= len(means) = len(stds))
    experiment_config = ExperimentConfig(
        time_horizon=1400,
        budget=300,
        price_range=list(range(5, 41, 5)),  # Prices from 5 to 40 with step 5
        num_trials=10,
        exploration_param=0.2,
        environment=env_config
    )
    
    return experiment_config


def create_multi_item_experiment_config() -> ExperimentConfig:
    """
    Example configuration for a multi-item experiment.
    
    Returns:
        Experiment configuration with 3 items
    """
    # Environment configuration for 3 different items
    # num_items = 3 is automatically inferred from list length
    env_config = EnvironmentConfig(
        means=[10, 15, 20],  # Different means for each item
        stds=[4, 5, 6],      # Different standard deviations for each item
    )
    
    # Experiment configuration doesn't specify num_items
    # It's automatically inferred as len(means) = len(stds) = 3
    experiment_config = ExperimentConfig(
        time_horizon=1000,
        budget=200,
        price_range=list(range(5, 31, 5)),  # Prices from 5 to 30
        num_trials=1,
        exploration_param=0.3,
        environment=env_config
    )
    
    return experiment_config


def create_environment_builder(env_config: EnvironmentConfig, 
                             time_horizon: int) -> callable:
    """
    Create a builder function for the environment.
    
    Args:
        env_config: Environment configuration
        time_horizon: Time horizon
        
    Returns:
        Function that creates new environment instances
    """
    def builder():
        return Environment(
            mean=env_config.means, 
            std=env_config.stds, 
            time_horizon=time_horizon
        )
    return builder


def create_agent_builder(experiment_config: ExperimentConfig) -> callable:
    """
    Create a builder function for the agent.
    
    Args:
        experiment_config: Experiment configuration
        
    Returns:
        Function that creates new agent instances
    """
    def builder():
        return CombinatorialUCBBidding(
            num_items=experiment_config.num_items,
            price_set=experiment_config.price_range,
            budget=experiment_config.budget,
            time_horizon=experiment_config.time_horizon,
            exploration_param=experiment_config.exploration_param
        )
    return builder


def print_experiment_summary(config: ExperimentConfig) -> None:
    """
    Print a summary of experiment parameters.

        Args:
        config: Experiment configuration
    """
    print("EXPERIMENT PARAMETERS:")
    print(f"- Time horizon: {config.time_horizon} rounds")
    print(f"- Total budget: {config.budget} units")
    print(f"- Number of items: {config.num_items}")
    print(f"- Available prices: {config.price_range}")
    print(f"- Number of trials: {config.num_trials}")
    print(f"- Exploration parameter: {config.exploration_param}")
    print(f"- Valuation means: {config.environment.means}")
    print(f"- Standard deviations: {config.environment.stds}")


def run_experiment_with_animation() -> None:
    """
    Execute simulation with animation generation.
    """
    print("Starting combinatorial pricing simulation with animation...")
    print("=" * 60)
    
    setup_reproducible_environment()
    
    experiment_config = create_multi_item_experiment_config()
    
    print(f"Number of items: {experiment_config.num_items}")
    print_experiment_summary(experiment_config)
    print("=" * 60)
    
    # Create builders
    env_builder = create_environment_builder(
        experiment_config.environment, 
        experiment_config.time_horizon
    )
    
    agent_builder = create_agent_builder(experiment_config)
    
    # Run one simulation with time series tracking for animation
    print("Running single simulation with time tracking for animation...")
    agent = agent_builder()
    env = env_builder()
    # TODO: qui facciamo una run a vuoto solo per generare una gif
    single_result = run_single_simulation(agent, env, experiment_config, track_time_series=True)
    
    # Create animated histogram
    if single_result.time_series_data:
        print("Generating animated histogram...")
        baseline_prices = np.array(single_result.baseline_price_per_item)
        create_animated_histogram(
            single_result.time_series_data,
            baseline_prices,
            experiment_config.price_range,
            output_filename="agent_prices.gif",
            fps=2
        )
    
    # Run full simulation for comprehensive results
    print("Running full simulation...")
    result = simulate(
        agent_builder=agent_builder,
        env_builder=env_builder,
        config=experiment_config,
    )
    
    # Visualize final results
    print("\nGenerating final plots...")
    plot_statistics(result, rounds_per_trial=experiment_config.time_horizon, 
                   n_trials=experiment_config.num_trials,
                   initial_budget=experiment_config.budget)
    
    print("Simulation with animation completed successfully!")


def run_experiment() -> None:
    """
    Execute the complete simulation experiment.
    """
    print("Starting combinatorial pricing simulation...")
    print("=" * 60)
    
    setup_reproducible_environment()
    
    experiment_config = create_multi_item_experiment_config()
    
    print(f"Number of items: {experiment_config.num_items}")
    print_experiment_summary(experiment_config)
    print("=" * 60)
    
    # Create builders
    env_builder = create_environment_builder(
        experiment_config.environment, 
        experiment_config.time_horizon
    )
    
    agent_builder = create_agent_builder(experiment_config)
    
    # Run simulation
    print("Starting simulation...")
    result = simulate(
        agent_builder=agent_builder,
        env_builder=env_builder,
        config=experiment_config,
    )
    
    # Plot configuration
    plotting_config = PlottingConfig(
        figure_size=(15, 10),
        dpi=150,
        save_plots=True,
        show_plots=True,
        output_filename="simulation_results.png"
    )
    
    # Visualize results
    print("\nGenerating comprehensive plots...")
    plot_statistics(result, rounds_per_trial=experiment_config.time_horizon, 
                   n_trials=experiment_config.num_trials,
                   initial_budget=experiment_config.budget)
    
    print("Simulation completed successfully!")


if __name__ == "__main__":
    run_experiment_with_animation()

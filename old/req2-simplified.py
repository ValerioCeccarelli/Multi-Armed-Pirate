import numpy as np

from src.agent import CombinatorialUCBBidding
from src.core import simulate, ExperimentConfig, EnvironmentConfig, run_single_simulation
from src.environment import Environment
from src.plotting import plot_statistics, create_animated_histogram

def get_env(env_config: EnvironmentConfig, experiment_config: ExperimentConfig) -> Environment:
    return Environment(
        mean=env_config.means, 
        std=env_config.stds, 
        time_horizon=experiment_config.time_horizon
    )

def get_agent(experiment_config: ExperimentConfig) -> CombinatorialUCBBidding:
    return CombinatorialUCBBidding(
        num_items=experiment_config.num_items,
        price_set=experiment_config.price_range,
        budget=experiment_config.budget,
        time_horizon=experiment_config.time_horizon,
        exploration_param=experiment_config.exploration_param
    )

def run_experiment_with_animation() -> None:
    env_config = EnvironmentConfig(
        means=[10, 15, 20],  # Different means for each item
        stds=[4, 5, 6],      # Different standard deviations for each item
    )

    experiment_config = ExperimentConfig(
        time_horizon=1200,
        budget=200,
        price_range=list(range(5, 31, 5)),  # Prices from 5 to 30
        num_trials=4,
        exploration_param=0.3,
        environment=env_config
    )

    single_result = run_single_simulation(get_agent(experiment_config), get_env(env_config, experiment_config), experiment_config, track_time_series=True)
    if single_result.time_series_data:
        baseline_prices = np.array(single_result.baseline_price_per_item)
        create_animated_histogram(
            single_result.time_series_data,
            baseline_prices,
            experiment_config.price_range,
            output_filename="agent_prices.gif",
            fps=2
        )
    
    result = simulate(
        agent_builder=lambda: get_agent(experiment_config),
        env_builder=lambda: get_env(env_config, experiment_config),
        config=experiment_config,
    )
    
    plot_statistics(result, rounds_per_trial=experiment_config.time_horizon, 
                   n_trials=experiment_config.num_trials,
                   initial_budget=experiment_config.budget)


if __name__ == "__main__":
    run_experiment_with_animation()

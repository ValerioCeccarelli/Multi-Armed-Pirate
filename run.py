from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from agents import Agent, UCBAgent
from environments import Environment, StochasticEnvironment
from plotting import (
    plot_budget_evolution,
    plot_cumulative_regret,
    plot_price_frequency_histograms,
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
        agent.update(rewards)

        total_played_arms[:, t] = price_indexes

    return RunSimulationResult(valuations=env.valuations, played_arms=total_played_arms)


if __name__ == "__main__":
    # Example usage with multiple trials
    num_items = 1
    num_prices = 3
    time_horizon = 100
    num_trials = 10
    prices = np.array([0.2, 0.5, 0.8])

    # Preallocate containers
    # valuations: (num_trials, num_items, time_horizon)
    valuations = np.empty((num_trials, num_items, time_horizon), dtype=np.float64)
    # agents_played_arms: (num_agents=1, num_trials, num_items, time_horizon)
    agents_played_arms = np.empty(
        (1, num_trials, num_items, time_horizon), dtype=np.int64
    )
    # baseline_played_arms: (num_trials, num_items, time_horizon)
    baseline_played_arms = np.empty(
        (num_trials, num_items, time_horizon), dtype=np.int64
    )

    for trial in range(num_trials):
        # New environment each trial to vary valuations across trials
        env = StochasticEnvironment(
            distribution_functions=[
                StochasticEnvironment.gaussian_distribution(mean=0.5, std=0.1)
            ],
            num_rounds=time_horizon,
        )

        # Agent and baseline for this trial (different seeds for reproducibility)
        agent = UCBAgent(num_prices=num_prices)
        baseline_agent = UCBAgent(num_prices=num_prices)

        result = run_simulation(env, agent, prices)
        baseline_results = run_simulation(env, baseline_agent, prices)

        # (num_items, time_horizon)
        valuations[trial] = result.valuations
        # (num_items, time_horizon)
        agents_played_arms[0, trial] = result.played_arms
        # (num_items, time_horizon)
        baseline_played_arms[trial] = baseline_results.played_arms

    # Plot price frequency histograms averaged across trials
    plot_budget_evolution(
        valuations=valuations,
        agents_played_arms=agents_played_arms,
        prices=prices,
        agents_names=["UCBAgent"],
        initial_budget=150,
    )

from dataclasses import dataclass
from typing import Callable, TypeVar
from numpy.typing import NDArray
import numpy as np
from matplotlib import pyplot as plt
import time

from environments import Environment
from agents import Agent


@dataclass
class RunSimulationResult:
    """
    Result of a single simulation run.

    Attributes:
        valuations: Valuations matrix (num_items, time_horizon)
        played_arms: Played arms matrix (num_items, time_horizon) (-1 if not played else arm_index)
    """

    valuations: NDArray[np.float64]
    played_arms: NDArray[np.int64]


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
        agent.update(rewards, valuations=valuations)

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

    valuations: NDArray[np.float64]
    agents_played_arms: NDArray[np.int64]


def run_multiple_simulations(
    env_builder: Callable[[], Environment],
    agent_builders: list[Callable[[Environment], Agent]],
    num_trials: int,
    prices: NDArray[np.float64],
) -> RunMultipleSimulationsResult:
    """
    Run multiple simulations of the agent interacting with the environment.

    Args:
        env_builder: A callable that returns a new Environment instance.
        agent_builder: A callable that takes the environment and returns a new Agent instance.
        num_trials: Number of simulation trials to run.
        prices: Prices array (num_prices,)

    Returns:
        RunMultipleSimulationsResult: The result of the simulations containing valuations and played arms.
    """

    # Initialize environment and agent to get dimensions
    temp_env = env_builder()  # TODO: This could be inefficient if env_builder is expensive
    num_items = temp_env.num_items
    time_horizon = temp_env.time_horizon
    num_agents = len(agent_builders)

    valuations = np.zeros(
        (num_trials, num_items, time_horizon), dtype=np.float64)

    agents_played_arms = np.full(
        (num_agents, num_trials, num_items, time_horizon), -1, dtype=np.int64
    )

    for trial in range(num_trials):
        print(f"Running trial {trial + 1}/{num_trials}...")
        env = env_builder()

        for i, agent_builder in enumerate(agent_builders):
            print(f"  Running agent {i + 1}/{num_agents}... ", end="")
            start = time.time()

            agent = agent_builder(env)
            agent_result = run_simulation(env, agent, prices)
            agents_played_arms[i, trial] = agent_result.played_arms

            end = time.time()
            print(f"Done in {end - start:.2f} seconds")

        valuations[trial] = env.valuations

    return RunMultipleSimulationsResult(
        valuations=valuations,
        agents_played_arms=agents_played_arms,
    )

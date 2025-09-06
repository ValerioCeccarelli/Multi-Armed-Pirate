from typing import Callable
from scipy.optimize import milp
from typing import Collection
from dataclasses import dataclass
import itertools

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize


class BudgetDepletedException(RuntimeError):
    pass


class Environment:
    def __init__(self, mean: list[float], std: list[float], T: int):
        # Valuations is (items, time) matrix
        # each line is the valuations for a specific item over time
        self.valuations = np.array(
            [np.random.normal(m, s, size=T) for m, s in zip(mean, std)]
        )

    def round(self, t: int, prices_t: list[float]) -> list[bool]:
        return [p <= v for p, v in zip(prices_t, self.valuations[:, t])]


@dataclass
class SimulationConfig:
    num_rounds: int
    prices_per_item: list[list[float]]
    total_budget: int


# TODO: scrivere che si tratta di un semi bandit feedback nella descrizione
@dataclass
class SingleSimulationResult:
    baseline_price_per_item: Collection[float]
    # Each dictionary maps price_idx to frequency
    arm_freq_per_item: Collection[dict[int, float]]
    # for each round, sum of rewards across all items
    agent_reward_sums: np.ndarray
    baseline_rewards: np.ndarray
    budget_depleted_round: int


@dataclass
class AggregatedSimulationResult:
    cumulative_regrets: np.ndarray
    # Each dictionary maps price_idx to mean of frequencies across trials
    mean_arm_freq_per_item: Collection[dict[int, float]]
    total_baseline_rewards: np.ndarray
    total_agent_rewards: np.ndarray
    budget_depleted_mean_round: float
    baseline_prices_per_item: np.ndarray


def get_baseline(env, config: SimulationConfig) -> tuple[Collection[float], np.ndarray]:
    """
    Return:
        best_prices: Collection[float] - the best prices for each item
        best_rewards: np.ndarray - the rewards at each time step respecting the budget
    """
    best_prices = []
    best_rewards = np.ndarray([])
    best_reward = -1

    # Create all possible combinations of prices for the items
    super_arms = list(itertools.product(*config.prices_per_item))

    for super_arm in super_arms:
        p = np.array(super_arm)
        # Create a mask where valuations are greater than or equal to the prices
        mask = env.valuations >= p[:, np.newaxis]

        # Calculate the rewards matrix
        rewards_matrix = mask * p[:, np.newaxis]

        # Calculate the cumulative sum of rewards over time
        cumulative_cost = np.cumsum(mask.sum(axis=0))

        # Apply the budget constraint
        budget_mask = cumulative_cost <= (
            config.total_budget - len(config.prices_per_item)
        )

        # Calculate the total reward for this super_arm
        total_rewards = rewards_matrix.sum(axis=0) * budget_mask
        total_reward = total_rewards.sum()

        if total_reward > best_reward:
            best_reward = total_reward
            best_prices = super_arm
            best_rewards = total_rewards

    return best_prices, best_rewards


class PrimalDualAgent:
    def __init__(self, num_items, price_set, B, T):
        self.N = num_items  # Number of item types (N)
        self.P = price_set  # Discrete set of possible prices for all items
        self.K_prices = len(self.P)  # Number of prices in the set

        self.T = T  # Time horizon
        self.t = 0

        self.B_initial = B  # Initial total budget
        self.remaining_budget = B

        # Stores the indices chosen in the last call to pull_arm
        self.last_chosen_indices = np.zeros(self.N, dtype=int)

    def pull_arm(self) -> np.ndarray:
        """
        Chooses one price index for each of the N items by solving an ILP.
        Returns a numpy array of price indices (one for each item).
        """
        raise NotImplementedError()

    def update(self, chosen_price_indices: np.ndarray, rewards: np.ndarray, costs: np.ndarray):
        """
        Updates agent statistics after a round using the chosen price indices.

        Args:
            chosen_price_indices (np.ndarray): Array of indices of prices chosen.
            rewards (np.ndarray): Array of rewards received for each item.
            costs (np.ndarray): Array of costs incurred for each item.
        """
        raise NotImplementedError()


def run_simulation(
    agent, env: Environment, config: SimulationConfig
) -> SingleSimulationResult:
    """Run one simulation episode for the combinatorial setting.
    Accumulates total (across items) rewards per round.
    """
    agent_rewards = []  # scalar per round (sum over items)

    # Baseline (best fixed super-arm) computed once on the true environment valuations
    baseline_prices_per_item, baseline_rewards = get_baseline(env, config)

    arm_freq_per_item = [dict() for _ in range(len(config.prices_per_item))]

    try:
        for t in range(config.num_rounds):
            # Stop if budget exhausted
            if agent.remaining_budget <= 0:
                raise BudgetDepletedException()

            price_indices = agent.pull_superarm()  # one index per item
            # Map indices to actual prices
            prices_t = [
                config.prices_per_item[i][idx] for i, idx in enumerate(price_indices)
            ]
            bought_list = env.round(t, prices_t)  # list[bool] length N
            rewards_items = np.array(
                [p if bought else 0 for p, bought in zip(
                    prices_t, bought_list)]
            )

            for i in range(len(config.prices_per_item)):
                if prices_t[i] not in arm_freq_per_item[i]:
                    arm_freq_per_item[i][prices_t[i]] = 0
                arm_freq_per_item[i][prices_t[i]] += 1

            costs_items = np.array(
                bought_list, dtype=int
            )  # each sale consumes 1 budget unit
            total_reward_t = rewards_items.sum()
            agent.update(price_indices, rewards_items, costs_items)
            agent_rewards.append(total_reward_t)
    except BudgetDepletedException:
        # Pad remaining rounds with zeros
        remaining = config.num_rounds - len(agent_rewards)
        if remaining > 0:
            agent_rewards.extend([0] * remaining)
    finally:
        budget_depleted_round = agent.t

    return SingleSimulationResult(
        baseline_price_per_item=baseline_prices_per_item,
        arm_freq_per_item=arm_freq_per_item,
        agent_reward_sums=np.array(agent_rewards),
        baseline_rewards=baseline_rewards,
        budget_depleted_round=budget_depleted_round,
    )


def simulate(
    agent_builder: Callable[[], PrimalDualAgent],
    env_builder: Callable[[], Environment],
    config: SimulationConfig,
    n_trials: int,
) -> AggregatedSimulationResult:
    cumulative_regrets = np.zeros((n_trials, config.num_rounds))
    total_baseline_rewards = np.zeros((n_trials, config.num_rounds))
    total_agent_rewards = np.zeros((n_trials, config.num_rounds))
    budget_depleted_rounds = np.zeros(n_trials, dtype=int)
    N_items = len(config.prices_per_item)
    baseline_prices_per_item = np.zeros((n_trials, N_items))
    mean_arm_freq_per_item = [dict() for _ in range(N_items)]

    for i in range(n_trials):
        print(f"Running trial {i+1}")
        agent = agent_builder()
        env = env_builder()
        sim_res = run_simulation(agent, env, config)

        regret = sim_res.baseline_rewards - sim_res.agent_reward_sums
        cumulative_regrets[i] = np.cumsum(regret)
        total_baseline_rewards[i] = sim_res.baseline_rewards
        total_agent_rewards[i] = sim_res.agent_reward_sums
        budget_depleted_rounds[i] = sim_res.budget_depleted_round
        baseline_prices_per_item[i] = np.array(sim_res.baseline_price_per_item)
        for i in range(N_items):
            for k, v in sim_res.arm_freq_per_item[i].items():
                if k not in mean_arm_freq_per_item[i]:
                    mean_arm_freq_per_item[i][k] = 0
                mean_arm_freq_per_item[i][k] += v

    for i in range(N_items):
        for k in mean_arm_freq_per_item[i]:
            mean_arm_freq_per_item[i][k] /= n_trials

    return AggregatedSimulationResult(
        cumulative_regrets=cumulative_regrets,
        mean_arm_freq_per_item=mean_arm_freq_per_item,
        total_baseline_rewards=total_baseline_rewards,
        total_agent_rewards=total_agent_rewards,
        budget_depleted_mean_round=budget_depleted_rounds.mean(),
        baseline_prices_per_item=baseline_prices_per_item,
    )


# --- Plotting utilities ---


def show_regret(cumulative_regrets: np.ndarray, rounds_per_trial: int, n_trials: int):
    avg_regret = cumulative_regrets.mean(axis=0)
    sd_regret = cumulative_regrets.std(axis=0)
    x = np.arange(rounds_per_trial)
    plt.plot(x, avg_regret, label="Average Regret")
    plt.fill_between(
        x,
        avg_regret - sd_regret / np.sqrt(n_trials),
        avg_regret + sd_regret / np.sqrt(n_trials),
        alpha=0.3,
        label="Uncertainty",
    )
    plt.xlabel("t")
    plt.title("Cumulative Regret")
    plt.legend()


def show_cumulative_rewards(
    total_baseline_rewards: np.ndarray,
    rounds_per_trial: int,
    total_agent_rewards: np.ndarray,
    budget_depleted_mean_round: float,
):
    cum_base = total_baseline_rewards.cumsum(axis=1)
    cum_agent = total_agent_rewards.cumsum(axis=1)
    x = np.arange(rounds_per_trial)
    plt.plot(x, cum_base.mean(axis=0), label="Average Baseline")
    plt.plot(x, cum_agent.mean(axis=0), label="Average Agent")
    plt.axvline(
        budget_depleted_mean_round,
        color="red",
        linestyle="--",
        label="Avg Budget Depletion",
    )
    plt.xlabel("t")
    plt.title("Cumulative Rewards")
    plt.legend()


def show_histogram(result: AggregatedSimulationResult):
    # Create separate histograms for each product/item based on mean_arm_freq_per_item
    n_items = len(result.mean_arm_freq_per_item)

    # Create subplots for each item
    fig, axes = plt.subplots(1, n_items, figsize=(5 * n_items, 4))

    # Handle single item case
    if n_items == 1:
        axes = [axes]

    baseline_price_per_item = result.baseline_prices_per_item.mean(axis=0)

    for item_idx in range(n_items):
        # Get the dictionary for this item
        item_freq_dict = result.mean_arm_freq_per_item[item_idx]

        if item_freq_dict:  # Check if dictionary is not empty
            # Extract rewards (keys) and frequencies (values)
            rewards = list(item_freq_dict.keys())
            frequencies = list(item_freq_dict.values())

            ax = axes[item_idx]
            ax.bar(rewards, frequencies, width=0.7, alpha=0.7)
            ax.set_xlabel("Reward")
            ax.set_ylabel("Mean Frequency")
            ax.set_title(f"Arm Frequencies for Item {item_idx + 1}")
            ax.grid(True, alpha=0.3)

            # Add best price line
            best_price = baseline_price_per_item[item_idx]
            ax.axvline(
                x=best_price,
                color="r",
                linestyle="--",
                label=f"Best Price: {best_price:.2f}",
            )
            ax.legend()
        else:
            # Handle empty dictionary case
            ax = axes[item_idx]
            ax.set_xlabel("Reward")
            ax.set_ylabel("Mean Frequency")
            ax.set_title(f"Arm Frequencies for Item {item_idx + 1} (No Data)")
            ax.grid(True, alpha=0.3)

            # Add best price line even for empty data
            best_price = baseline_price_per_item[item_idx]
            ax.axvline(
                x=best_price,
                color="r",
                linestyle="--",
                label=f"Best Price: {best_price:.2f}",
            )
            ax.legend()

    plt.tight_layout()


def plot_statistics(
    result: AggregatedSimulationResult, rounds_per_trial: int, n_trials: int
):
    print(
        f"Average budget depleted round: {result.budget_depleted_mean_round}")
    print(f"Average final regret: {result.cumulative_regrets[:, -1].mean()}")
    print(
        f"Mean of baseline best prices per item (averaged over trials): {result.baseline_prices_per_item.mean(axis=0)}"
    )
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    show_regret(result.cumulative_regrets, rounds_per_trial, n_trials)
    plt.subplot(2, 2, 2)
    show_cumulative_rewards(
        result.total_baseline_rewards,
        rounds_per_trial,
        result.total_agent_rewards,
        result.budget_depleted_mean_round,
    )
    plt.subplot(2, 2, 3)
    show_histogram(result)
    plt.tight_layout()
    plt.savefig("simulation_results.png", dpi=150, bbox_inches="tight")
    print("Plot saved as 'simulation_results.png'")
    plt.show()  # Comment out to avoid display issues


# --- Example experiment setup ---

n_trials = 10  # reduce for speed testing
T = 700  # reduce for speed testing
B = 300  # reduce for speed testing

# Environment builder: specify mean & std per item
means = [10, 15, 20]
stds = [5, 5, 5]

assert len(means) == len(stds)

N_items = len(means)
P = list(range(5, 31, 3))  # price set shared by all items
prices_per_item = [P for _ in range(N_items)]

simulation_config = SimulationConfig(
    num_rounds=T, prices_per_item=prices_per_item, total_budget=B
)


def env_builder(): return Environment(mean=means, std=stds, T=T)


def agent_builder(): return PrimalDualAgent(
    num_items=N_items, price_set=P, B=B, T=T, beta=0.2
)


result = simulate(
    agent_builder=agent_builder,
    env_builder=env_builder,
    config=simulation_config,
    n_trials=n_trials,
)
plot_statistics(result, rounds_per_trial=T, n_trials=n_trials)

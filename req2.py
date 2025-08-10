import itertools

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize


class BudgetDepletedException(RuntimeError):
    pass


class Environment:
    def __init__(self, mean: list[float | int], std: list[float | int], T: int):
        # Valuations is (items, time) matrix
        # each line is the valuations for a specific item over time
        self.valuations = np.array(
            [np.random.normal(m, s, size=T) for m, s in zip(mean, std)]
        )

    def round(self, t: int, prices_t: list[float | int]) -> list[bool]:
        return [p <= v for p, v in zip(prices_t, self.valuations[:, t])]


from dataclasses import dataclass
from typing import Collection


@dataclass
class SimulationConfig:
    num_rounds: int
    prices_per_item: list[list[float | int]]
    total_budget: int


# TODO: scrivere che si tratta di un semi bandit feedback nella descrizione
@dataclass
class SingleSimulationResult:
    best_price_per_item: Collection[float]
    agent_rewards: np.ndarray
    baseline_rewards: np.ndarray
    budget_depleted_round: int


@dataclass
class AggregatedSimulationResult:
    cumulative_regrets: np.ndarray
    total_baseline_rewards: np.ndarray
    total_agent_rewards: np.ndarray
    budget_depleted_mean_round: float
    best_prices_per_item: np.ndarray


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


import numpy as np
from scipy.optimize import milp


class CombinatorialUCBBidding:
    def __init__(self, num_items, price_set, B, T, beta=1.0):
        self.N = num_items  # Number of item types (N)
        self.P = price_set  # Discrete set of possible prices for all items
        self.K_prices = len(self.P)  # Number of prices in the set
        self.B_initial = B  # Initial total budget
        self.T = T  # Time horizon
        self.beta = beta  # UCB/LCB exploration parameter

        # Statistics are stored in N x K_prices matrices
        self.N_pulls = np.zeros((self.N, self.K_prices))
        self.avg_rewards = np.zeros((self.N, self.K_prices))
        self.avg_costs = np.zeros((self.N, self.K_prices))

        self.remaining_budget = B
        self.t = 0
        # Stores the indices chosen in the last call to pull_superarm
        self.last_chosen_indices = np.zeros(self.N, dtype=int)

    def pull_superarm(self) -> np.ndarray:
        """
        Chooses one price index for each of the N items by solving an ILP.
        Returns a numpy array of price indices.
        """
        # --- Exploration Phase ---
        # Ensure each arm (item, price combination) is pulled at least once.
        for n in range(self.N):
            for p_idx in range(self.K_prices):
                if self.N_pulls[n, p_idx] == 0:
                    # To explore arm (n, p_idx), select index p_idx for item n.
                    # For all other items, we can choose a default (e.g., index 0).
                    chosen_price_indices = np.zeros(self.N, dtype=int)
                    chosen_price_indices[n] = p_idx
                    self.last_chosen_indices = chosen_price_indices
                    return self.last_chosen_indices

        # --- UCB/LCB Calculation Phase ---
        # Add a small epsilon to avoid division by zero.
        confidence_term = np.sqrt((self.beta * np.log(self.T)) / (self.N_pulls + 1e-8))

        f_ucbs = self.avg_rewards + confidence_term
        c_lcbs = np.maximum(
            0, self.avg_costs - confidence_term
        )  # Costs cannot be negative

        # --- Optimization Phase (ILP) ---
        # Calculate the target spend rate for the remainder of the horizon.
        rho_t = self.remaining_budget / (self.T - self.t) if self.t < self.T else 0

        # Objective: Maximize sum(x_ij * f_ucbs_ij) which is equivalent to
        # Minimizing sum(x_ij * -f_ucbs_ij)
        # Decision variables x_ij are binary, 1 if we pick price j for item i.
        c = -f_ucbs.flatten()

        # Constraint 1 (Budget): Total expected cost must not exceed the spend rate.
        # sum(x_ij * c_lcbs_ij) <= rho_t
        A_ub = np.array([c_lcbs.flatten()])
        b_ub = np.array([rho_t])

        # Constraint 2 (Choice): Exactly one price must be chosen for each item.
        # For each item i, sum_j(x_ij) = 1
        A_eq = []
        for n in range(self.N):
            row = np.zeros(self.N * self.K_prices)
            row[n * self.K_prices : (n + 1) * self.K_prices] = 1
            A_eq.append(row)
        A_eq = np.array(A_eq)
        b_eq = np.ones(self.N)

        integrality = np.ones_like(c)  # All variables are integer (binary)
        bounds = (0, 1)  # All variables are between 0 and 1

        # Create LinearConstraint objects
        from scipy.optimize import LinearConstraint

        constraints = []
        if A_ub.size > 0:
            lb_ub = -1e20 * np.ones_like(
                b_ub
            )  # Use very large negative number instead of -inf
            constraints.append(LinearConstraint(A_ub, lb_ub, b_ub))
        if A_eq.size > 0:
            constraints.append(LinearConstraint(A_eq, b_eq, b_eq))

        # Solve the Integer Linear Program
        res = milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
        )

        if res.success and res.x is not None:
            # Extract solution: find the index of the '1' for each item row
            x_sol = np.round(res.x).reshape((self.N, self.K_prices))
            chosen_price_indices = np.argmax(x_sol, axis=1)
        else:
            # --- Fallback Strategy ---
            # If the solver fails, greedily pick the superarm with the highest UCB
            # reward among those whose LCB cost is below the spend rate.
            feasible_mask = c_lcbs <= rho_t
            masked_f_ucbs = np.where(feasible_mask, f_ucbs, -np.inf)
            chosen_price_indices = np.argmax(masked_f_ucbs, axis=1)

        self.last_chosen_indices = chosen_price_indices
        return self.last_chosen_indices

    def update(self, chosen_price_indices, rewards, costs):
        """
        Updates agent statistics after a round using the chosen price indices.

        Args:
            chosen_price_indices (np.ndarray): Array of indices of prices chosen.
            rewards (np.ndarray): Array of rewards received for each item.
            costs (np.ndarray): Array of costs incurred for each item.
        """
        # Iterate through each item and update the stats for the chosen price.
        for n in range(self.N):
            p_idx = chosen_price_indices[n]

            # Use incremental mean update formula: M_k = M_{k-1} + (x_k - M_{k-1}) / k
            # First, increment the pull count for the arm (n, p_idx).
            self.N_pulls[n, p_idx] += 1

            # Then, update the running averages for reward and cost.
            k = self.N_pulls[n, p_idx]
            self.avg_rewards[n, p_idx] += (rewards[n] - self.avg_rewards[n, p_idx]) / k
            self.avg_costs[n, p_idx] += (costs[n] - self.avg_costs[n, p_idx]) / k

        self.remaining_budget -= np.sum(costs)
        self.t += 1


from typing import Callable

import numpy as np

# --- Simulation utilities (multi-item adaptation of req1) ---


def run_simulation(
    agent, env: Environment, config: SimulationConfig
) -> SingleSimulationResult:
    """Run one simulation episode for the combinatorial setting.
    Accumulates total (across items) rewards per round.
    """
    agent_rewards = []  # scalar per round (sum over items)

    # Baseline (best fixed super-arm) computed once on the true environment valuations
    best_prices_per_item, baseline_rewards = get_baseline(env, config)

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
                [p if bought else 0 for p, bought in zip(prices_t, bought_list)]
            )
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
        best_price_per_item=best_prices_per_item,
        agent_rewards=np.array(agent_rewards),
        baseline_rewards=baseline_rewards,
        budget_depleted_round=budget_depleted_round,
    )


def simulate(
    agent_builder: Callable[[], CombinatorialUCBBidding],
    env_builder: Callable[[], Environment],
    config: SimulationConfig,
    n_trials: int,
) -> AggregatedSimulationResult:
    cumulative_regrets = np.zeros((n_trials, config.num_rounds))
    total_baseline_rewards = np.zeros((n_trials, config.num_rounds))
    total_agent_rewards = np.zeros((n_trials, config.num_rounds))
    budget_depleted_rounds = np.zeros(n_trials, dtype=int)
    N_items = len(config.prices_per_item)
    best_prices_per_item = np.zeros((n_trials, N_items))

    for i in range(n_trials):
        print(f"Running trial {i+1}")
        agent = agent_builder()
        env = env_builder()
        sim_res = run_simulation(agent, env, config)

        regret = sim_res.baseline_rewards - sim_res.agent_rewards
        cumulative_regrets[i] = np.cumsum(regret)
        total_baseline_rewards[i] = sim_res.baseline_rewards
        total_agent_rewards[i] = sim_res.agent_rewards
        budget_depleted_rounds[i] = sim_res.budget_depleted_round
        best_prices_per_item[i] = np.array(sim_res.best_price_per_item)

    return AggregatedSimulationResult(
        cumulative_regrets=cumulative_regrets,
        total_baseline_rewards=total_baseline_rewards,
        total_agent_rewards=total_agent_rewards,
        budget_depleted_mean_round=budget_depleted_rounds.mean(),
        best_prices_per_item=best_prices_per_item,
    )


# --- Plotting utilities ---
from matplotlib import pyplot as plt


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
    # Create separate histograms for each product/item
    n_items = result.best_prices_per_item.shape[1]

    # Create subplots for each item
    fig, axes = plt.subplots(1, n_items, figsize=(5 * n_items, 4))

    # Handle single item case
    if n_items == 1:
        axes = [axes]

    for item_idx in range(n_items):
        # Get prices for this specific item across all trials
        item_prices = result.best_prices_per_item[:, item_idx]
        unique, counts = np.unique(item_prices, return_counts=True)

        ax = axes[item_idx]
        ax.bar(unique, counts, width=0.7, alpha=0.7)
        ax.set_xlabel("Price")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Best Prices for Item {item_idx + 1}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()


def plot_statistics(
    result: AggregatedSimulationResult, rounds_per_trial: int, n_trials: int
):
    print(f"Average budget depleted round: {result.budget_depleted_mean_round}")
    print(f"Average final regret: {result.cumulative_regrets[:, -1].mean()}")
    print(
        f"Mean of baseline best prices per item (averaged over trials): {result.best_prices_per_item.mean(axis=0)}"
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
T = 500  # reduce for speed testing
B = 300  # reduce for speed testing
N_items = 3
P = list(range(5, 21, 5))  # price set shared by all items
prices_per_item = [P for _ in range(N_items)]

simulation_config = SimulationConfig(
    num_rounds=T, prices_per_item=prices_per_item, total_budget=B
)

# Environment builder: specify mean & std per item
means = [10, 15, 20]
stds = [4, 5, 6]

env_builder = lambda: Environment(mean=means, std=stds, T=T)

agent_builder = lambda: CombinatorialUCBBidding(
    num_items=N_items, price_set=P, B=B, T=T, beta=0.2
)

result = simulate(
    agent_builder=agent_builder,
    env_builder=env_builder,
    config=simulation_config,
    n_trials=n_trials,
)
plot_statistics(result, rounds_per_trial=T, n_trials=n_trials)

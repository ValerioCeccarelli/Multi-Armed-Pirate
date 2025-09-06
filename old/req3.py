import itertools
import random

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)


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


from dataclasses import dataclass
from typing import Collection


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


class PrimalDualBwRK:
    def __init__(self, num_items, prices_per_item, B, T, beta=0.0):
        self.N = num_items
        self.prices_per_item = prices_per_item
        self.B_initial = B
        self.T = T
        self.beta = beta  # Replenishment factor. beta=0 for standard BwK.

        # Super arms
        price_indices_per_item = [range(len(p)) for p in self.prices_per_item]
        self.super_arms_indices = list(itertools.product(*price_indices_per_item))
        self.num_super_arms = len(self.super_arms_indices)

        # Map from super arm tuple to index for weight update
        self.super_arm_to_idx = {
            arm: i for i, arm in enumerate(self.super_arms_indices)
        }

        # Normalization constants
        self.R_max = (
            self.N * max(self.prices_per_item[0]) if self.prices_per_item[0] else 1
        )
        self.C_max = self.N if self.N > 0 else 1

        self.rho = self.B_initial / self.T
        self.rho_norm = self.rho / self.C_max

        # For β=0, there's no replenishment from void action, but we need a void action.
        # Let's define it as choosing the highest price for all items.
        # The cost is expected to be 0, so c(∅) = 0, which means β=0.
        void_action_indices = []
        for p in self.prices_per_item:
            if p:
                void_action_indices.append(len(p) - 1)
            else:
                void_action_indices.append(0)  # or handle empty price list
        self.void_action_indices = tuple(void_action_indices)

        self.nu_norm = (self.beta / self.C_max) + self.rho_norm
        if self.nu_norm == 0:
            # Avoid division by zero, maybe agent should not be used
            self.lambda_max = 1e6
        else:
            self.lambda_max = 1.0 / self.nu_norm

        # Primal agent (EXP3)
        self.gamma_exp3 = np.sqrt(
            2 * np.log(self.num_super_arms) / (self.T * self.num_super_arms)
        )
        self.weights = np.ones(self.num_super_arms)
        self.p_t = np.ones(self.num_super_arms) / self.num_super_arms

        # Dual agent (OGD)
        self.lambda_ = 0.0
        # A simple learning rate for OGD.
        self.eta_ogd = 1.0 / np.sqrt(self.T)

        self.remaining_budget = float(B)
        self.t = 0
        self.last_chosen_indices = np.zeros(self.N, dtype=int)
        self.is_void_round = False
        self.budget_depleted_at = -1

        # For scaling primal utility
        # u'_t is in [-1/rho_norm + 1, 2] if f_t_norm is [0,1].
        # With beta=0, nu_norm=rho_norm.
        # u' in [f' + λ(ρ'-c')] = [0,1] + [0, 1/ρ'] * [ρ'-1, ρ'] = [0,1] + [1-1/ρ', 1]
        # min is 1-1/ρ', max is 2.
        self.u_p_norm_min = 1 - (1 / self.nu_norm) if self.nu_norm > 0 else 0
        self.u_p_norm_max = 2
        self.u_p_norm_range = self.u_p_norm_max - self.u_p_norm_min

    def pull_superarm(self) -> np.ndarray:
        if self.remaining_budget < 1:
            self.is_void_round = True
            if self.budget_depleted_at == -1:
                self.budget_depleted_at = self.t
            self.last_chosen_indices = np.array(self.void_action_indices)
            return self.last_chosen_indices

        self.is_void_round = False

        # Dual decision (already have self.lambda_ from previous update)

        # Primal decision (EXP3)
        self.p_t = (1 - self.gamma_exp3) * self.weights / np.sum(
            self.weights
        ) + self.gamma_exp3 / self.num_super_arms
        chosen_super_arm_idx = np.random.choice(self.num_super_arms, p=self.p_t)

        self.last_chosen_indices = np.array(
            self.super_arms_indices[chosen_super_arm_idx]
        )
        return self.last_chosen_indices

    def update(self, chosen_price_indices, rewards, costs):
        f_t = np.sum(rewards)
        c_t = np.sum(costs)

        self.remaining_budget -= c_t

        if not self.is_void_round:
            # Normalize
            f_t_norm = f_t / self.R_max
            c_t_norm = c_t / self.C_max

            # Primal update
            u_t_p = f_t_norm + self.lambda_ * (self.rho_norm - c_t_norm)

            # Scale to [0, 1] for EXP3
            if self.u_p_norm_range > 1e-6:
                u_t_p_scaled = (u_t_p - self.u_p_norm_min) / self.u_p_norm_range
            else:
                u_t_p_scaled = 0.5
            u_t_p_scaled = np.clip(u_t_p_scaled, 0, 1)

            chosen_super_arm_tuple = tuple(chosen_price_indices)
            chosen_super_arm_idx = self.super_arm_to_idx[chosen_super_arm_tuple]

            prob_chosen = self.p_t[chosen_super_arm_idx]
            r_hat = u_t_p_scaled / prob_chosen

            self.weights[chosen_super_arm_idx] *= np.exp(self.gamma_exp3 * r_hat)
            # Prevent weights from becoming too large
            if np.any(self.weights > 1e100):
                self.weights /= np.max(self.weights)

            # Dual update (OGD on negative utility)
            grad = self.rho_norm - c_t_norm
            self.lambda_ -= self.eta_ogd * grad
            self.lambda_ = np.clip(self.lambda_, 0, self.lambda_max)

        self.t += 1


from typing import Callable

# --- Simulation utilities (multi-item adaptation of req1) ---


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
                [p if bought else 0 for p, bought in zip(prices_t, bought_list)]
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
    agent_builder: Callable[[], PrimalDualBwRK],
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
        for item_idx in range(N_items):
            for k, v in sim_res.arm_freq_per_item[item_idx].items():
                if k not in mean_arm_freq_per_item[item_idx]:
                    mean_arm_freq_per_item[item_idx][k] = 0
                mean_arm_freq_per_item[item_idx][k] += v

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
    result: AggregatedSimulationResult,
    rounds_per_trial: int,
    n_trials: int,
    beta_value: float,
):
    print(f"Beta: {beta_value}")
    print(f"Average budget depleted round: {result.budget_depleted_mean_round}")
    print(f"Average final regret: {result.cumulative_regrets[:, -1].mean()}")
    print(
        f"Mean of baseline best prices per item (averaged over trials): {result.baseline_prices_per_item.mean(axis=0)}"
    )

    # Create the main plot with regret and rewards
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    show_regret(result.cumulative_regrets, rounds_per_trial, n_trials)
    plt.subplot(1, 2, 2)
    show_cumulative_rewards(
        result.total_baseline_rewards,
        rounds_per_trial,
        result.total_agent_rewards,
        result.budget_depleted_mean_round,
    )
    plt.suptitle(f"Performance Metrics (Beta = {beta_value})")
    plt.tight_layout()
    filename_performance = f"performance_beta_{beta_value}.png"
    plt.savefig(filename_performance, dpi=150, bbox_inches="tight")
    print(f"Performance plot saved as '{filename_performance}'")
    plt.close()  # Close to free memory

    # Create the histogram plot separately
    plt.figure(figsize=(15, 5))
    show_histogram(result)
    plt.suptitle(f"Arm Frequencies (Beta = {beta_value})")
    plt.tight_layout()
    filename_histogram = f"histogram_beta_{beta_value}.png"
    plt.savefig(filename_histogram, dpi=150, bbox_inches="tight")
    print(f"Histogram plot saved as '{filename_histogram}'")
    plt.close()  # Close to free memory


# --- Example experiment setup with hyperparameter tuning ---

# Environment builder: specify mean & std per item
# means = [10, 15, 20]
# stds = [4, 5, 6]
means = [10]
stds = [4]

n_trials = 10  # reduce for speed testing
T = 950  # reduce for speed testing
B = 300  # reduce for speed testing
N_items = len(means)
P = list(range(1, 21, 1))  # price set shared by all items
prices_per_item = [P for _ in range(N_items)]

simulation_config = SimulationConfig(
    num_rounds=T, prices_per_item=prices_per_item, total_budget=B
)


env_builder = lambda: Environment(mean=means, std=stds, T=T)

# Hyperparameter tuning for beta values
beta_values = [0.0]

print(f"Starting hyperparameter tuning for {len(beta_values)} beta values...")
print("=" * 60)

for beta in beta_values:
    print(f"\nRunning simulation with beta = {beta}")
    print("-" * 40)

    # Reset random seeds for each beta to ensure fair comparison
    random.seed(42)
    np.random.seed(42)

    agent_builder = lambda b=beta: PrimalDualBwRK(
        num_items=N_items, prices_per_item=prices_per_item, B=B, T=T, beta=b
    )

    result = simulate(
        agent_builder=agent_builder,
        env_builder=env_builder,
        config=simulation_config,
        n_trials=n_trials,
    )

    plot_statistics(result, rounds_per_trial=T, n_trials=n_trials, beta_value=beta)

print("\n" + "=" * 60)
print("Hyperparameter tuning completed!")

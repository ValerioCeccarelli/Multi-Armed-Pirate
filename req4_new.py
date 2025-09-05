
from dataclasses import dataclass
from typing import Callable
from matplotlib import pyplot as plt
import numpy as np
from src.core import TimeSeriesData, _aggregate_price_frequencies, _aggregate_success_failure_frequencies, _normalize_price_frequencies, _normalize_success_failure_frequencies, _update_price_frequencies, _update_success_failure_frequencies
from src.environment import AbruptSlightlyNonstationaryEnvironment, Environment
from src.agent import Agent, PrimalDualAgent, IntervalAwareBaselineAgent, BudgetDepletedException, CombinatorialUCBBidding
from src.plotting import plot_cumulative_regret, plot_cumulative_rewards

means = [[10, 30, 10], [40, 10, 40], [10, 10, 10], [5, 5, 5]]
stds = [[5,5,5], [5,5,5], [5,5,5], [5,5,5]]
time_horizon = 3000
num_items = len(means)

prices = list(range(5, 45, 4))
budget = time_horizon

num_trials=2

def env_builder():
    return AbruptSlightlyNonstationaryEnvironment(
        mean=means,
        std=stds,
        time_horizon=time_horizon
    )

def baseline_builder(env: Environment):
    return IntervalAwareBaselineAgent(
        num_items=num_items,
        time_horizon=time_horizon,
        price_set=prices,
        budget=budget,
        valuations=env.valuations,
    )

def agent_builder(env: Environment):
    return PrimalDualAgent(
        price_set=prices,
        num_items=num_items,
        B=budget,
        T=time_horizon,
        beta=0.1,
    )
    # return CombinatorialUCBBidding(
    #     price_set=prices,
    #     num_items=num_items,
    #     budget=budget,
    #     time_horizon=time_horizon,
    #     exploration_param=10,
    # )

@dataclass
class RunSimulationResult:
    """
    Result of a single simulation run.
    
    Attributes:
        budget_depleted_round (int): The round at which the budget was depleted.
        time_series_data (TimeSeriesData): Time series data of success and failure frequencies.
        price_frequencies (list[dict]): Frequency of each price being offered per item.
        success_frequencies (list[dict]): Frequency of successful purchases per price per item.
        failure_frequencies (list[dict]): Frequency of failed purchases per price per item.
        agent_rewards (list[float]): Rewards obtained by the agent at each round.
    """
    budget_depleted_round: int
    time_series_data: TimeSeriesData
    price_frequencies: list[dict[float, int]]
    success_frequencies: list[dict[float, int]]
    failure_frequencies: list[dict[float, int]]
    agent_rewards: list[float]
    agent_schedule: np.ndarray # shape (num_items, time_horizon)
    agent_taken: np.ndarray # shape (num_items, time_horizon) value 0/1

def run_simulation(agent, env) -> RunSimulationResult:
    num_items = env.num_items
    time_horizon = env.time_horizon
    prices = agent.price_set

    price_frequencies = [dict() for _ in range(num_items)]
    success_frequencies = [dict() for _ in range(num_items)]
    failure_frequencies = [dict() for _ in range(num_items)]

    success_over_time = []
    failure_over_time = []
    round_indices = []
    temp_success_freq = [dict() for _ in range(num_items)]
    temp_failure_freq = [dict() for _ in range(num_items)]

    agent_rewards = []  # Scalar reward per round (sum over all items)

    try:
        for t in range(time_horizon):
            # Agent interaction
            price_indexes = agent.select_prices()
            prices_t = [prices[i] for i in price_indexes]
            purchase_decisions = env.round(t, prices_t)

            rewards = np.array([
                price if purchased else 0 
                for price, purchased in zip(prices_t, purchase_decisions)
            ])
            costs = np.array(purchase_decisions, dtype=int)
            
            # TODO: agenti come PrimalDualAgent vogliono prices_t mentre CombinatorialUCBBidding price_indexes
            agent.update(prices_t, rewards, costs)

            # Track statistics
            _update_price_frequencies(price_frequencies, prices_t)
            _update_success_failure_frequencies(success_frequencies, failure_frequencies, prices_t, purchase_decisions)
            _update_success_failure_frequencies(temp_success_freq, temp_failure_freq, prices_t, purchase_decisions)

            if (t + 1) % 50 == 0 or t == time_horizon - 1:
                success_over_time.append([dict(freq) for freq in temp_success_freq])
                failure_over_time.append([dict(freq) for freq in temp_failure_freq])
                round_indices.append(t + 1)

            total_round_reward = rewards.sum()
            agent_rewards.append(total_round_reward)

    except BudgetDepletedException:
        # Complete remaining rounds with zero rewards
        remaining_rounds = time_horizon - len(agent_rewards)
        if remaining_rounds > 0:
            agent_rewards.extend([0] * remaining_rounds)
    finally:
        budget_depleted_round = agent.current_round
        
        time_series_data = TimeSeriesData(
            success_freq_over_time=success_over_time,
            failure_freq_over_time=failure_over_time,
            round_indices=round_indices
        )

    return RunSimulationResult(
        budget_depleted_round=budget_depleted_round,
        time_series_data=time_series_data,
        price_frequencies=price_frequencies,
        success_frequencies=success_frequencies,
        failure_frequencies=failure_frequencies,
        agent_rewards=agent_rewards,
        agent_schedule=agent.schedule,
        agent_taken=agent.taken,
    )

@dataclass
class RunMultipleSimulationsResult:
    """
    Result of multiple simulation runs.
    
    Attributes:
        cumulative_regrets: Cumulative regrets over time (num_trials, time_horizon).
        baseline_rewards_per_round: Rewards received by the baseline agent per round (num_trials, time_horizon).
        agent_rewards_per_round: Rewards received by the experimental agent per round (num_trials, time_horizon).ù
        agent_price_frequencies: Aggregated frequency of each price being offered per item by the experimental agent.
        agent_success_frequencies: Aggregated frequency of successful purchases per price per item by the experimental agent.
        agent_failure_frequencies: Aggregated frequency of failed purchases per price per item by the experimental agent.
        baseline_price_frequencies: Aggregated frequency of each price being offered per item by the baseline agent.
        baseline_success_frequencies: Aggregated frequency of successful purchases per price per item by the baseline agent.
        baseline_failure_frequencies: Aggregated frequency of failed purchases per price per item by the baseline agent.
        budget_depleted_rounds: Rounds at which the budget was depleted for each trial (num_trials,).
    """
    cumulative_regrets: np.ndarray # shape (num_trials, time_horizon)
    baseline_rewards_per_round: np.ndarray # shape (num_trials, time_horizon)
    agent_rewards_per_round: np.ndarray # shape (num_trials, time_horizon)

    agent_price_frequencies: list[dict[float, float]]
    agent_success_frequencies: list[dict[float, float]]
    agent_failure_frequencies: list[dict[float, float]]
    baseline_price_frequencies: list[dict[float, float]]
    baseline_success_frequencies: list[dict[float, float]]
    baseline_failure_frequencies: list[dict[float, float]]

    budget_depleted_rounds: np.ndarray # shape (num_trials,)

    agent_schedules: np.ndarray # shape (num_trials, num_items, time_horizon)
    baseline_schedules: np.ndarray # shape (num_trials, num_items, time_horizon)

    agent_taken: np.ndarray # shape (num_trials, num_items, time_horizon) value 0/1
    baseline_taken: np.ndarray # shape (num_trials, num_items, time_horizon) value 0/1


def run_multiple_simulations(agent_builder: Callable[[Environment], Agent], baseline_builder: Callable[[Environment], Agent], env_builder: Callable[[], Environment], num_trials: int) -> RunMultipleSimulationsResult:
    tmp_env = env_builder()
    time_horizon = tmp_env.time_horizon
    num_items = tmp_env.num_items

    agent_aggregated_price_frequencies = [dict() for _ in range(num_items)]
    agent_aggregated_success_frequencies = [dict() for _ in range(num_items)]
    agent_aggregated_failure_frequencies = [dict() for _ in range(num_items)]

    baseline_aggregated_price_frequencies = [dict() for _ in range(num_items)]
    baseline_aggregated_success_frequencies = [dict() for _ in range(num_items)]
    baseline_aggregated_failure_frequencies = [dict() for _ in range(num_items)]

    cumulative_regrets = np.zeros((num_trials, time_horizon))
    baseline_rewards_per_round = np.zeros((num_trials, time_horizon))
    agent_rewards_per_round = np.zeros((num_trials, time_horizon))
    budget_depleted_rounds = np.zeros(num_trials, dtype=int)

    agent_schedules = np.zeros((num_trials, num_items, time_horizon), dtype=int)
    baseline_schedules = np.zeros((num_trials, num_items, time_horizon), dtype=int)

    agent_taken = np.zeros((num_trials, num_items, time_horizon), dtype=int)
    baseline_taken = np.zeros((num_trials, num_items, time_horizon), dtype=int)

    # Run multiple simulations
    for trial_idx in range(num_trials):
        print(f"Running trial {trial_idx + 1}/{num_trials}")
        env = env_builder()

        agent = agent_builder(env)
        baseline = baseline_builder(env)

        agent_result = run_simulation(agent, env)
        baseline_result = run_simulation(baseline, env)

        regret_per_round = np.array(baseline_result.agent_rewards) - np.array(agent_result.agent_rewards)
        cumulative_regrets[trial_idx] = np.cumsum(regret_per_round)

        baseline_rewards_per_round[trial_idx] = baseline_result.agent_rewards
        agent_rewards_per_round[trial_idx] = agent_result.agent_rewards

        budget_depleted_rounds[trial_idx] = agent_result.budget_depleted_round

        agent_schedules[trial_idx] = agent_result.agent_schedule
        baseline_schedules[trial_idx] = baseline_result.agent_schedule

        agent_taken[trial_idx] = agent_result.agent_taken
        baseline_taken[trial_idx] = baseline_result.agent_taken

        _aggregate_price_frequencies(agent_aggregated_price_frequencies, agent_result.price_frequencies, num_items)
        _aggregate_success_failure_frequencies(agent_aggregated_success_frequencies, agent_result.success_frequencies, num_items)
        _aggregate_success_failure_frequencies(agent_aggregated_failure_frequencies, agent_result.failure_frequencies, num_items)

        _aggregate_price_frequencies(baseline_aggregated_price_frequencies, baseline_result.price_frequencies, num_items)
        _aggregate_success_failure_frequencies(baseline_aggregated_success_frequencies, baseline_result.success_frequencies, num_items)
        _aggregate_success_failure_frequencies(baseline_aggregated_failure_frequencies, baseline_result.failure_frequencies, num_items)

    _normalize_price_frequencies(agent_aggregated_price_frequencies, num_trials, num_items)
    _normalize_success_failure_frequencies(agent_aggregated_success_frequencies, num_trials, num_items)
    _normalize_success_failure_frequencies(agent_aggregated_failure_frequencies, num_trials, num_items)

    _normalize_price_frequencies(baseline_aggregated_price_frequencies, num_trials, num_items)
    _normalize_success_failure_frequencies(baseline_aggregated_success_frequencies, num_trials, num_items)
    _normalize_success_failure_frequencies(baseline_aggregated_failure_frequencies, num_trials, num_items)

    return RunMultipleSimulationsResult(
        cumulative_regrets=cumulative_regrets,
        baseline_rewards_per_round=baseline_rewards_per_round,
        agent_rewards_per_round=agent_rewards_per_round,
        agent_price_frequencies=agent_aggregated_price_frequencies,
        agent_success_frequencies=agent_aggregated_success_frequencies,
        agent_failure_frequencies=agent_aggregated_failure_frequencies,
        baseline_price_frequencies=baseline_aggregated_price_frequencies,
        baseline_success_frequencies=baseline_aggregated_success_frequencies,
        baseline_failure_frequencies=baseline_aggregated_failure_frequencies,
        budget_depleted_rounds=budget_depleted_rounds,
        agent_schedules=agent_schedules,
        baseline_schedules=baseline_schedules,
        agent_taken=agent_taken,
        baseline_taken=baseline_taken,
    )

results = run_multiple_simulations(agent_builder, baseline_builder, env_builder, num_trials)

def plot_price_frequency_histograms(
        success_freq: list[dict[float, float]], 
        failure_freq: list[dict[float, float]],
        title: str = "Price Frequency Histograms",
        save_path: str = None):
    num_items = len(success_freq)

    fig, axes = plt.subplots(1, num_items, figsize=(5 * num_items, 4))
    # Ensure axes is always iterable
    try:
        len(axes)
    except TypeError:
        axes = [axes]

    max_frequency = 0
    all_data = []

    for item_idx in range(num_items):
        success_frequencies = success_freq[item_idx]
        failure_frequencies = failure_freq[item_idx]

        all_prices = set(success_frequencies.keys()) | set(failure_frequencies.keys())

        if all_prices:
            prices = sorted(list(all_prices))
            successes = [success_frequencies.get(price, 0) for price in prices]
            failures = [failure_frequencies.get(price, 0) for price in prices]
            
            # Track max frequency for y-axis alignment
            max_frequency = max(max_frequency, max(max(successes), max(failures)))
            
            all_data.append((prices, successes, failures, all_prices))
        else:
            all_data.append(([], [], [], set()))

    # Set a common y-limit with some padding
    y_max = max_frequency * 2 if max_frequency > 0 else 1
    # TODO: y_max is not working or is not the right value

    for item_idx in range(num_items):
        current_ax = axes[item_idx]
        prices, successes, failures, all_prices = all_data[item_idx]

        # Calculate bar positions for stacked display
        bar_width = 0.6
        x_positions = np.arange(len(prices))
        
        # Create stacked bars (failures on bottom, successes on top)
        current_ax.bar(x_positions, successes, bar_width,
                        label='Successful Purchases', color='green', alpha=0.7)
        current_ax.bar(x_positions, failures, bar_width, bottom=successes,
                        label='Failed Purchases', color='pink', alpha=1)
        
        # Set x-axis labels to show actual prices
        current_ax.set_xticks(x_positions)
        current_ax.set_xticklabels([f"{p:.0f}" for p in prices])

        # Set common y-axis limits for all subplots
        current_ax.set_ylim(0, y_max)
        current_ax.set_xlabel("Price")
        current_ax.set_ylabel("Mean Frequency")
        current_ax.set_title(f"Purchase Outcomes - Item {item_idx + 1}")
        current_ax.grid(True, alpha=0.3)
        current_ax.legend()

    # Layout, title, save, and show once for the whole figure
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    # Do not call plt.show() here to avoid blocking; caller will decide when to show
    return fig

def plot_conversion_rates(
        success_freqs: list[dict[float, float]], 
        failure_freqs: list[dict[float, float]],
        ) -> None:
    ax = plt.gca()
    
    num_items = len(success_freqs)
    
    # Colors for different items
    colors = plt.cm.Set1(np.linspace(0, 1, num_items))
    
    for item_idx in range(num_items):
        success_freq = success_freqs[item_idx]
        failure_freq = failure_freqs[item_idx]
        
        # Calculate conversion rates
        prices = []
        conversion_rates = []
        
        all_prices = set(success_freq.keys()) | set(failure_freq.keys())
        
        for price in sorted(all_prices):
            successes = success_freq.get(price, 0)
            failures = failure_freq.get(price, 0)
            total_attempts = successes + failures
            
            if total_attempts > 0:
                conversion_rate = (successes / total_attempts) * 100
                prices.append(price)
                conversion_rates.append(conversion_rate)
        
        # Plot conversion rate line
        ax.plot(prices, conversion_rates, 'o-', linewidth=2, markersize=8,
                label=f'Item {item_idx + 1}', color=colors[item_idx])
            
    ax.set_xlabel("Price")
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_title("Price Conversion Rates by Item")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 105)

def plot_budget_evolution(
        budget_depleted_round: float,
        agent_rewards: np.ndarray,
        initial_budget: int,
):
    print(f"Average budget depletion round: {budget_depleted_round}")
    ax = plt.gca()

    num_trials, num_rounds = agent_rewards.shape

    # Estimate budget spent based on successful purchases
    # This is an approximation since we don't track exact budget in results
    cumulative_purchases = np.zeros((num_trials, num_rounds))

    # TODO: use costs instead of rewards
    for trial_idx in range(num_trials):
        purchases_per_round = (agent_rewards[trial_idx] > 0).astype(int)
        cumulative_purchases[trial_idx] = np.cumsum(purchases_per_round)

    mean_cumulative_purchases = cumulative_purchases.mean(axis=0)
    std_cumulative_purchases = cumulative_purchases.std(axis=0)
    
    rounds = np.arange(num_rounds)
    remaining_budget = initial_budget - mean_cumulative_purchases
    
    ax.plot(rounds, remaining_budget, label="Average Remaining Budget", linewidth=2, color='blue')
    ax.fill_between(rounds, 
                   initial_budget - (mean_cumulative_purchases + std_cumulative_purchases),
                   initial_budget - (mean_cumulative_purchases - std_cumulative_purchases),
                   alpha=0.3, color='blue', label="Budget Range")
    
    # Add budget depletion line
    ax.axvline(x=budget_depleted_round, color='red', linestyle='--', 
              linewidth=2, label=f'Avg Depletion Round: {budget_depleted_round:.0f}')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    ax.set_xlabel("Round")
    ax.set_ylabel("Remaining Budget")
    ax.set_title("Budget Evolution Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_summary_metrics(
        cumulative_regrets: np.ndarray,
        baseline_rewards: np.ndarray,
        agent_rewards: np.ndarray,
        depleted_budget_rounds: np.ndarray,
        initial_budget: int,
        save_path: str = None):
    n_trials, rounds_per_trial = cumulative_regrets.shape

    fig = plt.figure(figsize=(18, 6))

    # Subplot 1: Cumulative regret
    plt.subplot(1, 3, 1)
    plot_cumulative_regret(cumulative_regrets, rounds_per_trial, n_trials)

    # Subplot 2: Cumulative rewards
    plt.subplot(1, 3, 2)
    plot_cumulative_rewards(
        baseline_rewards,
        agent_rewards,
        depleted_budget_rounds.mean(),
        rounds_per_trial
    )

    # Subplot 3: Conversion rates
    # plt.subplot(2, 2, 3)
    # plot_conversion_rates(
    #     results.agent_success_frequencies,
    #     results.agent_failure_frequencies,
    # )

    # Subplot 4: Budget evolution
    plt.subplot(1, 3, 3)
    plot_budget_evolution(
        depleted_budget_rounds.mean(),
        agent_rewards,
        initial_budget=initial_budget
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    # Do not call plt.show() here to avoid blocking; caller will decide when to show
    return fig


agent_schedule = results.agent_schedules[0]
agent_taken = results.agent_taken[0]

def plot_histogram_per_interval(schedule: np.ndarray, taken: np.ndarray, num_intervals: int):
    """
    Plot a single figure with a grid of subplots shaped (num_intervals x num_items).
    Each row corresponds to an interval of rounds; each column to an item. Bars are
    stacked (success on top of failure) and aligned by price.
    """
    num_items, time_horizon = schedule.shape
    rounds_per_interval = time_horizon // num_intervals

    # Create a single figure for all intervals and items
    fig, axesx = plt.subplots(num_intervals, num_items, figsize=(5 * num_items, 4 * num_intervals))

    # Ensure axesx is always a 2D array for consistent indexing
    if num_intervals == 1 and num_items == 1:
        axesx = np.array([[axesx]])
    elif num_intervals == 1:
        axesx = np.array([axesx])
    elif num_items == 1:
        axesx = np.array([[ax] for ax in axesx])

    for interval_index in range(num_intervals):
        start_round = interval_index * rounds_per_interval
        end_round = (interval_index + 1) * rounds_per_interval if interval_index < num_intervals - 1 else time_horizon

        interval_schedule = schedule[:, start_round:end_round]
        interval_taken = taken[:, start_round:end_round]

        for item_idx in range(num_items):
            current_ax = axesx[interval_index, item_idx]
            item_prices = interval_schedule[item_idx]
            item_taken = interval_taken[item_idx]

            success_dict: dict = {}
            failure_dict: dict = {}

            for price, taken_flag in zip(item_prices, item_taken):
                if taken_flag == 1:
                    success_dict[price] = success_dict.get(price, 0) + 1
                else:
                    failure_dict[price] = failure_dict.get(price, 0) + 1

            # remove -1 keys if present
            success_dict.pop(-1, None)
            failure_dict.pop(-1, None)

            # Align x across success/failure keys
            all_prices = sorted(set(success_dict.keys()) | set(failure_dict.keys()))
            success_vals = [success_dict.get(p, 0) for p in all_prices]
            failure_vals = [failure_dict.get(p, 0) for p in all_prices]

            # plot histogram where success and failure are stacked
            bar_width = 0.6
            current_ax.bar(all_prices, success_vals, width=bar_width,
                           label='Successful Purchases', color='green', alpha=0.7)
            current_ax.bar(all_prices, failure_vals, width=bar_width,
                           label='Failed Purchases', color='pink', alpha=1, bottom=success_vals)

            current_ax.set_xlabel("Price")
            current_ax.set_ylabel("Frequency")
            current_ax.set_title(f"Item {item_idx + 1} - Rounds {start_round + 1} to {end_round}")
            current_ax.legend()

    plt.tight_layout()
    return fig

plot_histogram_per_interval(agent_schedule, agent_taken, num_intervals=len(means[0]))

    

# plot_price_frequency_histograms(
#     results.agent_success_frequencies,
#     results.agent_failure_frequencies,
# )

# plot_price_frequency_histograms(
#     results.baseline_success_frequencies,
#     results.baseline_failure_frequencies,
# )

plot_summary_metrics(
    results.cumulative_regrets,
    results.baseline_rewards_per_round,
    results.agent_rewards_per_round,
    results.budget_depleted_rounds,
    initial_budget=budget,
)

# # Show all figures at once (single blocking call)
plt.show()


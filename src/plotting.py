import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from .core import AggregatedSimulationResult


@dataclass
class PlottingConfig:
    """
    Configuration for visualization plots.
    
    Attributes:
        figure_size: Figure dimensions (width, height)
        dpi: Resolution for saving
        save_plots: Whether to save plots to file
        show_plots: Whether to display plots on screen
        output_filename: Output filename for plots
    """
    figure_size: Tuple[int, int] = (15, 10)
    dpi: int = 150
    save_plots: bool = True
    show_plots: bool = True
    output_filename: str = "simulation_results.png"
    
    def __post_init__(self):
        """Validate plot configuration."""
        if self.figure_size[0] <= 0 or self.figure_size[1] <= 0:
            raise ValueError("Figure dimensions must be positive")
        if self.dpi <= 0:
            raise ValueError("DPI must be positive")
        if not self.output_filename:
            raise ValueError("Output filename cannot be empty")


def plot_cumulative_regret(cumulative_regrets: np.ndarray, rounds_per_trial: int, 
                          n_trials: int, ax: Optional[plt.Axes] = None) -> None:
    """
    Visualize mean cumulative regret with confidence bands.
    
    Args:
        cumulative_regrets: Matrix of cumulative regrets (trials x rounds)
        rounds_per_trial: Number of rounds per trial
        n_trials: Number of trials
        ax: Optional matplotlib axis for plotting
    """
    if ax is None:
        ax = plt.gca()
    
    average_regret = cumulative_regrets.mean(axis=0)
    std_regret = cumulative_regrets.std(axis=0)
    standard_error = std_regret / np.sqrt(n_trials)
    
    rounds = np.arange(rounds_per_trial)
    
    ax.plot(rounds, average_regret, label="Average Regret", linewidth=2)
    ax.fill_between(
        rounds,
        average_regret - standard_error,
        average_regret + standard_error,
        alpha=0.3,
        label="Confidence Interval",
    )
    
    ax.set_xlabel("Round (t)")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title("Cumulative Regret Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_cumulative_rewards(total_baseline_rewards: np.ndarray,
                           total_agent_rewards: np.ndarray,
                           budget_depleted_mean_round: float,
                           rounds_per_trial: int,
                           ax: Optional[plt.Axes] = None) -> None:
    """
    Visualize cumulative rewards of agent vs baseline.
    
    Args:
        total_baseline_rewards: Baseline rewards for each trial
        total_agent_rewards: Agent rewards for each trial
        budget_depleted_mean_round: Mean round of budget depletion
        rounds_per_trial: Number of rounds per trial
        ax: Optional matplotlib axis for plotting
    """
    if ax is None:
        ax = plt.gca()
    
    cumulative_baseline = total_baseline_rewards.cumsum(axis=1)
    cumulative_agent = total_agent_rewards.cumsum(axis=1)
    rounds = np.arange(rounds_per_trial)
    
    ax.plot(rounds, cumulative_baseline.mean(axis=0), 
           label="Average Baseline", linewidth=2, color='green')
    ax.plot(rounds, cumulative_agent.mean(axis=0), 
           label="Average Agent", linewidth=2, color='blue')
    
    ax.axvline(
        budget_depleted_mean_round,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Average Budget Depletion (Round {budget_depleted_mean_round:.1f})",
    )
    
    ax.set_xlabel("Round (t)")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Rewards: Agent vs Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_price_frequency_histograms(result: AggregatedSimulationResult,
                                   ax: Optional[plt.Axes] = None) -> None:
    """
    Visualize histograms of price frequencies for each item.
    Shows green bars for successful purchases and red bars for failed purchases.
    Y-axes are aligned across all items for easy comparison.
    
    Args:
        result: Aggregated simulation results
        ax: Optional matplotlib axis for plotting
    """
    num_items = len(result.mean_arm_success_freq_per_item)
    
    if ax is None:
        fig, axes = plt.subplots(1, num_items, figsize=(6 * num_items, 5))
        if num_items == 1:
            axes = [axes]
    else:
        axes = [ax] if num_items == 1 else [ax]

    baseline_prices = result.baseline_prices_per_item.mean(axis=0)
    
    # Calculate global y-axis limits for alignment
    max_frequency = 0
    all_data = []
    
    for item_idx in range(num_items):
        success_frequencies = result.mean_arm_success_freq_per_item[item_idx]
        failure_frequencies = result.mean_arm_failure_freq_per_item[item_idx]
        
        # Get all unique prices that were tried
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
    y_max = max_frequency * 1.1 if max_frequency > 0 else 1

    for item_idx in range(num_items):
        current_ax = axes[item_idx] if num_items > 1 else axes[0]
        prices, successes, failures, all_prices = all_data[item_idx]
        
        if all_prices:
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
            
            # Add optimal price line
            optimal_price = baseline_prices[item_idx]
            if prices:  # Only if we have prices to compare
                closest_price_idx = min(range(len(prices)), 
                                      key=lambda i: abs(prices[i] - optimal_price))
                current_ax.axvline(
                    x=closest_price_idx,
                    color="black",
                    linestyle="--",
                    linewidth=2,
                    label=f"Optimal Price: {optimal_price:.2f}",
                )
        else:
            # Handle empty data case
            optimal_price = baseline_prices[item_idx]
            current_ax.text(0.5, 0.5, 'No Data', transform=current_ax.transAxes,
                           ha='center', va='center', fontsize=12)

        # Set common y-axis limits for all subplots
        current_ax.set_ylim(0, y_max)
        current_ax.set_xlabel("Price")
        current_ax.set_ylabel("Mean Frequency")
        current_ax.set_title(f"Purchase Outcomes - Item {item_idx + 1}")
        current_ax.grid(True, alpha=0.3)
        current_ax.legend()

    if ax is None:
        plt.tight_layout()


def plot_conversion_rates(result: AggregatedSimulationResult, ax: Optional[plt.Axes] = None) -> None:
    """
    Plot conversion rates (success percentage) for each price and item.
    
    Args:
        result: Aggregated simulation results
        ax: Optional matplotlib axis for plotting
    """
    if ax is None:
        ax = plt.gca()
    
    num_items = len(result.mean_arm_success_freq_per_item)
    baseline_prices = result.baseline_prices_per_item.mean(axis=0)
    
    # Colors for different items
    colors = plt.cm.Set1(np.linspace(0, 1, num_items))
    
    for item_idx in range(num_items):
        success_freq = result.mean_arm_success_freq_per_item[item_idx]
        failure_freq = result.mean_arm_failure_freq_per_item[item_idx]
        
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
        
        if prices:
            # Plot conversion rate line
            ax.plot(prices, conversion_rates, 'o-', linewidth=2, markersize=8,
                   label=f'Item {item_idx + 1}', color=colors[item_idx])
            
            # Add optimal price marker
            optimal_price = baseline_prices[item_idx]
            ax.axvline(x=optimal_price, color=colors[item_idx], linestyle='--', alpha=0.7,
                      label=f'Optimal Price Item {item_idx + 1}: {optimal_price:.1f}')
    
    ax.set_xlabel("Price")
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_title("Price Conversion Rates by Item")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 105)


def plot_budget_evolution(result: AggregatedSimulationResult, initial_budget: int, 
                         ax: Optional[plt.Axes] = None) -> None:
    """
    Plot budget depletion over time.
    
    Args:
        result: Aggregated simulation results
        initial_budget: Initial budget amount
        ax: Optional matplotlib axis for plotting
    """
    if ax is None:
        ax = plt.gca()
    
    # Calculate cumulative purchases (budget spent)
    num_trials, num_rounds = result.total_agent_rewards.shape
    
    # Estimate budget spent based on successful purchases
    # This is an approximation since we don't track exact budget in results
    cumulative_purchases = np.zeros((num_trials, num_rounds))
    
    # TODO: use costs instead of rewards
    for trial_idx in range(num_trials):
        purchases_per_round = (result.total_agent_rewards[trial_idx] > 0).astype(int)
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
    ax.axvline(x=result.budget_depleted_mean_round, color='red', linestyle='--', 
              linewidth=2, label=f'Avg Depletion Round: {result.budget_depleted_mean_round:.0f}')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    ax.set_xlabel("Round")
    ax.set_ylabel("Remaining Budget")
    ax.set_title("Budget Evolution Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_performance_metrics(result: AggregatedSimulationResult, ax: Optional[plt.Axes] = None) -> None:
    """
    Plot key performance metrics as a summary dashboard.
    
    Args:
        result: Aggregated simulation results
        ax: Optional matplotlib axis for plotting
    """
    if ax is None:
        ax = plt.gca()
    
    # Calculate metrics
    final_regret = result.cumulative_regrets[:, -1].mean()
    regret_std = result.cumulative_regrets[:, -1].std()
    
    total_agent_reward = result.total_agent_rewards.sum(axis=1).mean()
    total_baseline_reward = result.total_baseline_rewards.sum(axis=1).mean()
    
    efficiency = (total_agent_reward / total_baseline_reward) * 100 if total_baseline_reward > 0 else 0
    
    # Create bar chart of metrics
    metrics = ['Final Regret', 'Agent Reward', 'Baseline Reward', 'Efficiency (%)']
    values = [final_regret, total_agent_reward, total_baseline_reward, efficiency]
    colors = ['red', 'blue', 'green', 'orange']
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
               f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title("Performance Metrics Summary")
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


def generate_comprehensive_plots(result: AggregatedSimulationResult, 
                               rounds_per_trial: int, 
                               n_trials: int,
                               initial_budget: int = 300,
                               config: PlottingConfig = None) -> None:
    """
    Generate a comprehensive set of plots to analyze simulation results.
    
    Args:
        result: Aggregated simulation results
        rounds_per_trial: Number of rounds per trial
        n_trials: Number of trials executed
        initial_budget: Initial budget for budget evolution plot
        config: Optional plot configuration
    """
    if config is None:
        config = PlottingConfig()
    
    # Print summary statistics
    print_simulation_summary(result)
    
    # Create main figure with 2x2 layout for comprehensive analysis
    fig = plt.figure(figsize=(12, 12))
    
    # Subplot 1: Cumulative regret
    plt.subplot(2, 2, 1)
    plot_cumulative_regret(result.cumulative_regrets, rounds_per_trial, n_trials)
    
    # Subplot 2: Cumulative rewards
    plt.subplot(2, 2, 2)
    plot_cumulative_rewards(
        result.total_baseline_rewards,
        result.total_agent_rewards,
        result.budget_depleted_mean_round,
        rounds_per_trial
    )
    
    # Subplot 3: Conversion rates
    plt.subplot(2, 2, 3)
    plot_conversion_rates(result)
    
    # Subplot 4: Budget evolution
    plt.subplot(2, 2, 4)
    plot_budget_evolution(result, initial_budget)
    
    plt.tight_layout()
    
    if config.save_plots:
        plt.savefig(config.output_filename, dpi=config.dpi, bbox_inches="tight")
        print(f"Plot saved as '{config.output_filename}'")
    
    if config.show_plots:
        plt.show()

        # Histograms of price frequencies
        plot_price_frequency_histograms(result)
        plt.show()


def print_simulation_summary(result: AggregatedSimulationResult) -> None:
    """
    Print a textual summary of simulation results.
    
    Args:
        result: Aggregated simulation results
    """
    print("=" * 60)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Average budget depletion round: {result.budget_depleted_mean_round:.2f}")
    print(f"Average final regret: {result.cumulative_regrets[:, -1].mean():.2f}")
    print(f"Mean baseline prices per item: {result.baseline_prices_per_item.mean(axis=0)}")
    print("=" * 60)


# Aliases for compatibility with existing interface
def show_regret(cumulative_regrets: np.ndarray, rounds_per_trial: int, n_trials: int):
    """Alias for plot_cumulative_regret for compatibility."""
    plot_cumulative_regret(cumulative_regrets, rounds_per_trial, n_trials)


def show_cumulative_rewards(total_baseline_rewards: np.ndarray,
                           rounds_per_trial: int,
                           total_agent_rewards: np.ndarray,
                           budget_depleted_mean_round: float):
    """Alias for plot_cumulative_rewards for compatibility."""
    plot_cumulative_rewards(total_baseline_rewards, total_agent_rewards,
                           budget_depleted_mean_round, rounds_per_trial)


def show_histogram(result: AggregatedSimulationResult):
    """Alias for plot_price_frequency_histograms for compatibility."""
    plot_price_frequency_histograms(result)


def create_animated_histogram(time_series_data, baseline_prices: np.ndarray, 
                             price_range: List[float], output_filename: str = "agent_prices.gif",
                             fps: int = 2) -> None:
    """
    Create an animated GIF showing the evolution of price frequency histograms over time.
    
    Args:
        time_series_data: TimeSeriesData containing evolution over time
        baseline_prices: Optimal baseline prices for each item
        price_range: Available price range
        output_filename: Output filename for the GIF
        fps: Frames per second for the animation
    """
    if not time_series_data or not time_series_data.success_freq_over_time:
        print("No time series data available for animation")
        return
    
    num_items = len(time_series_data.success_freq_over_time[0])
    num_frames = len(time_series_data.round_indices)
    
    # Calculate global y-axis limits for consistent scaling
    max_frequency = 0
    for frame_idx in range(num_frames):
        for item_idx in range(num_items):
            success_freq = time_series_data.success_freq_over_time[frame_idx][item_idx]
            failure_freq = time_series_data.failure_freq_over_time[frame_idx][item_idx]
            
            for freq_dict in [success_freq, failure_freq]:
                if freq_dict:
                    max_frequency = max(max_frequency, max(freq_dict.values()))
    
    y_max = max_frequency * 1.1 if max_frequency > 0 else 1
    
    # Set up the figure and axes
    fig, axes = plt.subplots(1, num_items, figsize=(6 * num_items, 5))
    if num_items == 1:
        axes = [axes]
    
    # Initialize empty bars for each item
    bars_success = []
    bars_failure = []
    
    for item_idx in range(num_items):
        ax = axes[item_idx] if num_items > 1 else axes[0]
        
        # Create initial empty bars
        x_positions = np.arange(len(price_range))
        bar_width = 0.6
        
        bar_success = ax.bar(x_positions, [0] * len(price_range), bar_width,
                           label='Successful Purchases', color='green', alpha=0.7)
        bar_failure = ax.bar(x_positions, [0] * len(price_range), bar_width, 
                           bottom=[0] * len(price_range),
                           label='Failed Purchases', color='pink', alpha=1)
        
        bars_success.append(bar_success)
        bars_failure.append(bar_failure)
        
        # Set up axes
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{p:.0f}" for p in price_range])
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Price")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Purchase Outcomes - Item {item_idx + 1}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add optimal price line
        optimal_price = baseline_prices[item_idx]
        if price_range:
            closest_price_idx = min(range(len(price_range)), 
                                  key=lambda i: abs(price_range[i] - optimal_price))
            ax.axvline(x=closest_price_idx, color="black", linestyle="--", 
                      linewidth=2, label=f"Optimal: {optimal_price:.2f}")
    
    def animate(frame_idx):
        """Animation function for each frame."""
        round_num = time_series_data.round_indices[frame_idx]
        
        for item_idx in range(num_items):
            success_freq = time_series_data.success_freq_over_time[frame_idx][item_idx]
            failure_freq = time_series_data.failure_freq_over_time[frame_idx][item_idx]
            
            # Get frequencies for each price in the range
            successes = [success_freq.get(price, 0) for price in price_range]
            failures = [failure_freq.get(price, 0) for price in price_range]
            
            # Update bar heights
            for i, (bar_s, bar_f) in enumerate(zip(bars_success[item_idx], bars_failure[item_idx])):
                bar_s.set_height(successes[i])
                bar_f.set_height(failures[i])
                bar_f.set_y(successes[i])  # Stack failures on top of successes
        
        # Update title with current round
        fig.suptitle(f"Price Learning Evolution - Round {round_num}", fontsize=16)
        
        return [bar for bars in bars_success + bars_failure for bar in bars]
    
    # Create animation
    print(f"Creating animated histogram with {num_frames} frames...")
    anim = FuncAnimation(fig, animate, frames=num_frames, interval=1000/fps, 
                        blit=True, repeat=True)
    
    # Save as GIF
    # TODO: Uncomment to enable saving
    # print(f"Saving animation as '{output_filename}'...")
    # writer = PillowWriter(fps=fps)
    # anim.save(output_filename, writer=writer)
    # print(f"Animation saved successfully!")
    
    plt.close(fig)


def plot_statistics(result: AggregatedSimulationResult, rounds_per_trial: int, n_trials: int, initial_budget: int = 300):
    """Alias for generate_comprehensive_plots for compatibility."""
    generate_comprehensive_plots(result, rounds_per_trial, n_trials, initial_budget) 
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from matplotlib.animation import FuncAnimation


def plot_cumulative_regret(
    valuations: NDArray[np.float64],
    agents_played_arms: NDArray[np.int64],
    baseline_played_arms: NDArray[np.int64],
    prices: NDArray[np.float64],
    agents_names: list[str],
    title: str = "Cumulative Regret Over Time",
    ax: plt.Axes = None,
    save_plot: bool = False,
    save_path: str = "cumulative_regret.png",
) -> None:
    """Plot cumulative regret over time.

    Args:
        valuations: Valuations matrix (num_trials, num_items, time_horizon)
        agents_played_arms: Played arms by agents (num_agents, num_trials,
            num_items, time_horizon)
        baseline_played_arms: Played arms by baseline (num_trials, num_items,
            time_horizon)
        prices: Prices matrix (num_prices,)
        agents_names: Names of the agents (num_agents,)
        title: Title of the plot
        ax: Matplotlib Axes to plot on. If None, creates a new figure and axes and
            plots there.
        save_plot: Whether to save the plot to a file.
        save_path: Path to save the plot to.
    """

    num_agents, num_trials, num_items, time_horizon = agents_played_arms.shape
    assert valuations.shape == (num_trials, num_items, time_horizon), (
        f"Expected valuations shape {(num_trials, num_items, time_horizon)}, got"
        f" {valuations.shape}"
    )
    assert baseline_played_arms.shape == (num_trials, num_items, time_horizon), (
        "Expected baseline_played_arms shape"
        f" {(num_trials, num_items, time_horizon)}, got"
        f" {baseline_played_arms.shape}"
    )
    assert (
        len(agents_names) == num_agents
    ), f"Expected {num_agents} agent names, got {len(agents_names)}"

    is_new_figure = ax is None
    if is_new_figure:
        fig, ax = plt.subplots(figsize=(10, 6))

    for agent_idx in range(num_agents):
        # (num_trials, num_items, time_horizon)
        agent_played_prices = prices[agents_played_arms[agent_idx]]
        # (num_trials, num_items, time_horizon)
        baseline_played_prices = prices[baseline_played_arms]

        # (num_trials, num_items, time_horizon)
        agent_rewards = np.where(
            valuations >= agent_played_prices, agent_played_prices, 0.0
        )
        # (num_trials, num_items, time_horizon)
        baseline_rewards = np.where(
            valuations >= baseline_played_prices, baseline_played_prices, 0.0
        )

        average_agent_rewards = np.mean(
            np.sum(agent_rewards, axis=1), axis=0
        )  # (time_horizon,)
        average_baseline_rewards = np.mean(
            np.sum(baseline_rewards, axis=1), axis=0
        )  # (time_horizon,)

        std_agent_rewards = np.std(
            np.sum(agent_rewards, axis=1), axis=0
        )  # (time_horizon,)
        std_baseline_rewards = np.std(
            np.sum(baseline_rewards, axis=1), axis=0
        )  # (time_horizon,)

        cumulative_regrets = np.cumsum(
            average_baseline_rewards - average_agent_rewards
        )  # (time_horizon,)
        cumulative_std = np.sqrt(
            np.cumsum(std_agent_rewards**2 + std_baseline_rewards**2)
        )  # (time_horizon,)

        ax.plot(cumulative_regrets, label=agents_names[agent_idx])
        ax.fill_between(
            np.arange(time_horizon),
            cumulative_regrets - cumulative_std,
            cumulative_regrets + cumulative_std,
            alpha=0.3,
            label=f"{agents_names[agent_idx]} confidence interval",
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Regret")
    ax.legend()

    if is_new_figure:
        if save_plot:
            plt.savefig(save_path)
        plt.show()


def plot_animated_price_frequency_histograms(
    valuations: NDArray[np.float64],
    agents_played_arms: NDArray[np.int64],
    prices: NDArray[np.float64],
    agents_names: list[str],
    save_path_prefix: str = "animated_price_frequency",
) -> None:
    """Creates animated price frequency histograms for each agent and item.

    Args:
        valuations: Valuations matrix (num_trials, num_items, time_horizon)
        agents_played_arms: Played arms by agents (num_agents, num_trials, num_items, time_horizon)
        prices: Prices matrix (num_prices,)
        agents_names: Names of the agents (num_agents,)
        save_path_prefix: Prefix for the path to save the plots to. Plots will be
            saved as f"{save_path_prefix}_{agent_name}.gif".
    """

    num_agents, num_trials, num_items, time_horizon = agents_played_arms.shape
    assert valuations.shape == (num_trials, num_items, time_horizon), (
        f"Expected valuations shape {(num_trials, num_items, time_horizon)}, got"
        f" {valuations.shape}"
    )
    assert (
        len(agents_names) == num_agents
    ), f"Expected {num_agents} agent names, got {len(agents_names)}"

    for agent_idx in range(num_agents):
        fig, axes = plt.subplots(
            1, num_items, figsize=(6 * num_items, 5), squeeze=False
        )
        axes = axes.flatten()
        fig.suptitle(
            f"Animated Price Frequency Histograms for {agents_names[agent_idx]}",
            fontsize=16,
        )

        def update(frame):
            for item_idx in range(num_items):
                axes[item_idx].cla()
                axes[item_idx].set_title(f"Item {item_idx + 1} (Time: {min(frame * 10 + 1, time_horizon)})")
                axes[item_idx].set_xlabel("Price")
                axes[item_idx].set_ylabel("Mean Frequency")
                
                # Calculate successful and failed purchases up to current frame
                end_frame = min(frame * 10 + 1, time_horizon)
                arm_idx_to_num_successful_pulls = dict()
                arm_idx_to_num_failed_pulls = dict()
                
                for trial_idx in range(num_trials):
                    for t in range(end_frame):
                        curr_valuation = valuations[trial_idx, item_idx, t]
                        played_arm = agents_played_arms[agent_idx, trial_idx, item_idx, t]
                        
                        if played_arm == -1:
                            break  # Budget exhausted
                            
                        if curr_valuation >= prices[played_arm]:
                            arm_idx_to_num_successful_pulls[played_arm] = (
                                arm_idx_to_num_successful_pulls.get(played_arm, 0) + 1
                            )
                        else:
                            arm_idx_to_num_failed_pulls[played_arm] = (
                                arm_idx_to_num_failed_pulls.get(played_arm, 0) + 1
                            )
                
                # Calculate mean frequencies across trials
                success_freq = {
                    arm_idx: num_successful_pulls / num_trials
                    for arm_idx, num_successful_pulls in arm_idx_to_num_successful_pulls.items()
                }
                failure_freq = {
                    arm_idx: num_failed_pulls / num_trials
                    for arm_idx, num_failed_pulls in arm_idx_to_num_failed_pulls.items()
                }
                
                all_arms = set(success_freq.keys()) | set(failure_freq.keys())
                if all_arms:
                    sorted_arms = sorted(list(all_arms))
                    successes = np.array([success_freq.get(arm, 0) for arm in sorted_arms])
                    failures = np.array([failure_freq.get(arm, 0) for arm in sorted_arms])
                    bar_prices = [prices[arm] for arm in sorted_arms]
                    
                    bar_width = 0.05
                    x_positions = np.array(bar_prices)
                    
                    # Create stacked bars like in the static plot
                    axes[item_idx].bar(
                        x_positions,
                        successes,
                        bar_width,
                        label="Successful Purchases",
                        color="green",
                        alpha=0.7,
                    )
                    axes[item_idx].bar(
                        x_positions,
                        failures,
                        bar_width,
                        bottom=successes,
                        label="Failed Purchases",
                        color="pink",
                        alpha=1,
                    )
                    
                    # Set consistent y-axis limits
                    max_height = np.max(successes + failures) if len(successes) > 0 else 1
                    axes[item_idx].set_ylim(0, max_height * 1.1)
                
                axes[item_idx].set_xlim(prices[0] - 0.05, prices[-1] + 0.05)
                axes[item_idx].grid(True, alpha=0.3)
                axes[item_idx].legend()
                
            return axes

        # Adjust frames to be reasonable for the animation
        total_frames = min(100, time_horizon // 10)
        ani = FuncAnimation(fig, update, frames=total_frames, interval=200,
                            blit=False, repeat=True)

        # Save the animation 
        save_path = f"{save_path_prefix}_{agents_names[agent_idx]}.gif"
        ani.save(save_path, writer='pillow', fps=5)
        print(f"Animazione salvata in: {save_path}")
        
        plt.show()


def plot_price_frequency_histograms(
    valuations: NDArray[np.float64],
    agents_played_arms: NDArray[np.int64],
    prices: NDArray[np.float64],
    agents_names: list[str],
    save_plot: bool = False,
    save_path_prefix: str = "price_frequency",
) -> None:
    """Plots price frequency histograms for each agent and item.

    Args:
        valuations: Valuations matrix (num_trials, num_items, time_horizon)
        agents_played_arms: Played arms by agents (num_agents, num_trials, num_items, time_horizon)
        prices: Prices matrix (num_prices,)
        agents_names: Names of the agents (num_agents,)
        save_plot: Whether to save the plot to a file.
        save_path_prefix: Prefix for the path to save the plots to. Plots will be
            saved as f"{save_path_prefix}_{agent_name}.png".
    """
    num_agents, num_trials, num_items, time_horizon = agents_played_arms.shape
    assert valuations.shape == (num_trials, num_items, time_horizon), (
        f"Expected valuations shape {(num_trials, num_items, time_horizon)}, got"
        f" {valuations.shape}"
    )
    assert (
        len(agents_names) == num_agents
    ), f"Expected {num_agents} agent names, got {len(agents_names)}"

    for agent_idx in range(num_agents):
        agent_data = []
        max_frequency = 0
        for item_idx in range(num_items):
            arm_idx_to_num_successful_pulls = dict()
            arm_idx_to_num_failed_pulls = dict()
            for trial_idx in range(num_trials):
                for t in range(time_horizon):
                    curr_valuation = valuations[trial_idx, item_idx, t]
                    played_arm = agents_played_arms[agent_idx,
                                                    trial_idx, item_idx, t]
                    if played_arm == -1:
                        # Reached round where budget was exhausted, stop
                        break
                    if curr_valuation >= prices[played_arm]:
                        arm_idx_to_num_successful_pulls[played_arm] = (
                            arm_idx_to_num_successful_pulls.get(
                                played_arm, 0) + 1
                        )
                    else:
                        arm_idx_to_num_failed_pulls[played_arm] = (
                            arm_idx_to_num_failed_pulls.get(played_arm, 0) + 1
                        )
            # Calculate the mean number of successful/failed pulls per arm across
            # all trials
            success_freq = {
                arm_idx: num_successful_pulls / num_trials
                for arm_idx, num_successful_pulls in arm_idx_to_num_successful_pulls.items()
            }
            failure_freq = {
                arm_idx: num_failed_pulls / num_trials
                for arm_idx, num_failed_pulls in arm_idx_to_num_failed_pulls.items()
            }

            all_arms = set(success_freq.keys()) | set(failure_freq.keys())
            if all_arms:
                sorted_arms = sorted(list(all_arms))
                successes = np.array([success_freq.get(arm, 0)
                                     for arm in sorted_arms])
                failures = np.array([failure_freq.get(arm, 0)
                                    for arm in sorted_arms])
                bar_prices = [prices[arm] for arm in sorted_arms]
                max_frequency = max(
                    max_frequency, np.max(successes + failures))
                agent_data.append(
                    {
                        "arms": sorted_arms,
                        "prices": bar_prices,
                        "successes": successes,
                        "failures": failures,
                    }
                )
            else:
                agent_data.append(None)

        y_max = max_frequency * 1.1 if max_frequency > 0 else 1
        fig, axes = plt.subplots(
            1, num_items, figsize=(6 * num_items, 5), squeeze=False
        )
        axes = axes.flatten()
        fig.suptitle(
            f"Price Frequency Histograms for {agents_names[agent_idx]}", fontsize=16
        )

        for item_idx in range(num_items):
            ax = axes[item_idx]
            item_data = agent_data[item_idx]
            if item_data:
                bar_width = 0.6
                x_positions = np.arange(len(item_data["arms"]))
                ax.bar(
                    x_positions,
                    item_data["successes"],
                    bar_width,
                    label="Successful Purchases",
                    color="green",
                    alpha=0.7,
                )
                ax.bar(
                    x_positions,
                    item_data["failures"],
                    bar_width,
                    bottom=item_data["successes"],
                    label="Failed Purchases",
                    color="pink",
                    alpha=1,
                )
                ax.set_xticks(x_positions)
                ax.set_xticklabels([f"{p:.2f}" for p in item_data["prices"]])
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                )

            ax.set_ylim(0, y_max)
            ax.set_xlabel("Price")
            ax.set_ylabel("Mean Frequency")
            ax.set_title(f"Item {item_idx + 1}")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # adjust for suptitle
        if save_plot:
            plt.savefig(f"{save_path_prefix}_{agents_names[agent_idx]}.png")
        # plt.show()


def plot_budget_evolution(
    valuations: NDArray[np.float64],
    agents_played_arms: NDArray[np.int64],
    prices: NDArray[np.float64],
    agents_names: list[str],
    initial_budget: np.int64,
    ax: plt.Axes = None,
    save_plot: bool = False,
    save_path_prefix: str = "budget_evolution",
):
    """Plots the evolution of budgets over time for each agent and item.

    Args:
        valuations: Valuations matrix (num_trials, num_items, time_horizon)
        agents_played_arms: Played arms by agents (num_agents, num_trials, num_items, time_horizon)
        prices: Prices matrix (num_prices,)
        agents_names: Names of the agents (num_agents,)
        initial_budget: Initial budget
        ax: Matplotlib Axes to plot on. If None, creates a new figure and axes and
            plots there.
        save_plot: Whether to save the plot to a file.
        save_path_prefix: Prefix for the path to save the plots to. Plots will be
            saved as f"{save_path_prefix}_{agent_name}.png".
    """
    num_agents, num_trials, num_items, time_horizon = agents_played_arms.shape
    assert valuations.shape == (num_trials, num_items, time_horizon), (
        f"Expected valuations shape {(num_trials, num_items, time_horizon)}, got"
        f" {valuations.shape}"
    )
    assert (
        len(agents_names) == num_agents
    ), f"Expected {num_agents} agent names, got {len(agents_names)}"

    is_new_figure = ax is None
    if is_new_figure:
        fig, ax = plt.subplots(figsize=(10, 6))

    for agent_idx in range(num_agents):
        remaining_budget = np.zeros((num_trials, time_horizon + 1))
        remaining_budget[:, 0] = initial_budget
        for t in range(time_horizon):
            depleted_budget = np.zeros(num_trials)
            for item_idx in range(num_items):
                played_arms = agents_played_arms[agent_idx, :, item_idx, t]
                valuations_t = valuations[:, item_idx, t]
                valid_pulls = played_arms != -1
                played_prices = np.where(valid_pulls, prices[played_arms], 0)
                successful_purchases = np.where(
                    valid_pulls & (played_prices <= valuations_t), 1, 0
                )
                depleted_budget += successful_purchases
            remaining_budget[:, t + 1] = remaining_budget[:,
                                                          t] - depleted_budget

        mean_remaining_budget = np.mean(remaining_budget, axis=0)
        std_remaining_budget = np.std(remaining_budget, axis=0)

        ax.plot(
            range(time_horizon + 1),
            mean_remaining_budget,
            label=agents_names[agent_idx],
        )
        ax.fill_between(
            range(time_horizon + 1),
            mean_remaining_budget - std_remaining_budget,
            mean_remaining_budget + std_remaining_budget,
            alpha=0.3,
            label=f"{agents_names[agent_idx]} confidence interval",
        )

    ax.set_title("Budget Evolution Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Remaining Budget")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if is_new_figure:
        if save_plot:
            plt.savefig(f"{save_path_prefix}_budget_evolution.png")
        # plt.show()

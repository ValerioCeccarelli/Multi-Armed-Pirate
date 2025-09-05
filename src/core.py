from typing import Callable, Collection, Tuple, List, Dict, Optional
from dataclasses import dataclass
import itertools
import numpy as np

from .agent import CombinatorialUCBBidding, BudgetDepletedException
from .environment import Environment


@dataclass
class EnvironmentConfig:
    """
    Configuration for the simulation environment.
    
    Attributes:
        means: Valuation means for each item
        stds: Valuation standard deviations for each item
    """
    means: List[float]
    stds: List[float]
    
    @property
    def num_items(self) -> int:
        """Number of configured items."""
        return len(self.means)


@dataclass
class ExperimentConfig:
    """
    Complete configuration for a simulation experiment.
    
    Attributes:
        time_horizon: Number of time rounds to simulate
        budget: Total budget available for the agent
        price_range: List of available prices for all items
        num_trials: Number of trials to execute
        exploration_param: UCB exploration parameter (beta)
        environment: Environment configuration
    """
    time_horizon: int
    budget: int
    price_range: List[float]
    num_trials: int
    exploration_param: float
    environment: EnvironmentConfig
    num_intervals: int = -1
    
    def __post_init__(self):
        """Validate the experiment configuration."""
        if self.time_horizon <= 0:
            raise ValueError("Time horizon must be positive")
        if self.budget <= 0:
            raise ValueError("Budget must be positive")
        if not self.price_range:
            raise ValueError("Price range cannot be empty")
        if self.num_trials <= 0:
            raise ValueError("Number of trials must be positive")
        if self.exploration_param < 0:
            raise ValueError("Exploration parameter must be non-negative")
    
    @property
    def num_items(self) -> int:
        """Number of items inferred from environment configuration."""
        return self.environment.num_items
    
    @property
    def prices_per_item(self) -> List[List[float]]:
        """Generate prices per item list for compatibility."""
        return [self.price_range for _ in range(self.num_items)]


@dataclass
class TimeSeriesData:
    """
    Time series data for animation purposes.
    
    Attributes:
        success_freq_over_time: Success frequencies for each item over time
        failure_freq_over_time: Failure frequencies for each item over time
        round_indices: Round indices corresponding to the data points
    """
    success_freq_over_time: List[List[Dict[float, int]]]  # [round][item][price] -> count
    failure_freq_over_time: List[List[Dict[float, int]]]  # [round][item][price] -> count
    round_indices: List[int]


@dataclass
class SingleSimulationResult:
    """
    Result of a single simulation.
    
    Note: This involves semi-bandit feedback in the description
    """
    baseline_price_per_item: Collection[float]
    arm_freq_per_item: Collection[Dict[int, float]]  # Maps price_idx -> frequency
    arm_success_freq_per_item: Collection[Dict[float, int]]  # Maps price -> successful purchases
    arm_failure_freq_per_item: Collection[Dict[float, int]]  # Maps price -> failed purchases
    agent_reward_sums: np.ndarray  # Sum of rewards for each round
    baseline_rewards: np.ndarray
    budget_depleted_round: int
    time_series_data: Optional[TimeSeriesData] = None  # For animation


@dataclass
class AggregatedSimulationResult:
    """
    Aggregated result of multiple simulations.
    
    Attributes:
        cumulative_regrets: Cumulative regret for each trial
        mean_arm_freq_per_item: Mean frequency of arms for each item
        mean_arm_success_freq_per_item: Mean frequency of successful purchases per price
        mean_arm_failure_freq_per_item: Mean frequency of failed purchases per price
        total_baseline_rewards: Total baseline rewards
        total_agent_rewards: Total agent rewards
        budget_depleted_mean_round: Mean round of budget depletion
        baseline_prices_per_item: Baseline prices for each item
    """
    cumulative_regrets: np.ndarray
    mean_arm_freq_per_item: Collection[Dict[int, float]]
    mean_arm_success_freq_per_item: Collection[Dict[float, float]]
    mean_arm_failure_freq_per_item: Collection[Dict[float, float]]
    total_baseline_rewards: np.ndarray
    total_agent_rewards: np.ndarray
    budget_depleted_mean_round: float
    baseline_prices_per_item: np.ndarray


def calculate_baseline_performance(env: Environment, 
                                 config: ExperimentConfig) -> Tuple[Collection[float], np.ndarray]:
    """
    Calculate optimal baseline performance for the given environment.
    
    The baseline represents the best possible fixed pricing strategy,
    calculated with perfect knowledge of future valuations.
    
    Args:
        env: Simulation environment
        config: Experiment configuration
        
    Returns:
        Tuple containing:
        - best_prices: Optimal prices for each item
        - best_rewards: Rewards for each timestep respecting budget
    """
    best_prices = []
    best_rewards = np.array([])
    best_total_reward = -1

    # Generate all possible price combinations for items
    all_price_combinations = list(itertools.product(*config.prices_per_item))

    for price_combination in all_price_combinations:
        prices_array = np.array(price_combination)
        
        # Calculate performance for this price combination
        total_reward, rewards_per_round = _evaluate_price_combination(
            env, prices_array, config.budget, config.num_items
        )

        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_prices = price_combination
            best_rewards = rewards_per_round

    return best_prices, best_rewards


def _evaluate_price_combination(env: Environment, prices: np.ndarray, 
                               total_budget: int, num_items: int) -> Tuple[float, np.ndarray]:
    """
    Evaluate a specific price combination.
    
    Args:
        env: Simulation environment
        prices: Array of prices for each item
        total_budget: Total available budget
        num_items: Number of items
        
    Returns:
        Tuple of (total reward, rewards per round)
    """
    # Mask where valuations are >= prices (purchases)
    purchase_mask = env.valuations >= prices[:, np.newaxis]
    
    # Rewards matrix (price * purchase)
    rewards_matrix = purchase_mask * prices[:, np.newaxis]
    
    # Cumulative cost over time (number of purchases)
    cumulative_purchases = np.cumsum(purchase_mask.sum(axis=0))
    
    # Apply budget constraint
    budget_constraint = cumulative_purchases <= (total_budget - num_items)
    
    # Calculate total rewards for each round
    rewards_per_round = rewards_matrix.sum(axis=0) * budget_constraint
    total_reward = rewards_per_round.sum()
    
    return total_reward, rewards_per_round


def run_single_simulation(agent: CombinatorialUCBBidding, env: Environment, 
                         config: ExperimentConfig, track_time_series: bool = False) -> SingleSimulationResult:
    """
    Run a single simulation episode for the combinatorial setting.
    
    Args:
        agent: UCB bidding agent
        env: Simulation environment
        config: Experiment configuration
        
    Returns:
        Single simulation result
    """
    agent_rewards = []  # Scalar reward per round (sum over all items)
    
    # Calculate baseline once using true environment valuations
    baseline_prices, baseline_rewards = calculate_baseline_performance(env, config)
    
    # Track frequency of price usage for each item
    price_frequencies = [dict() for _ in range(config.num_items)]
    success_frequencies = [dict() for _ in range(config.num_items)]
    failure_frequencies = [dict() for _ in range(config.num_items)]
    
    # Track time series data for animation if requested
    time_series_data = None
    if track_time_series:
        success_over_time = []
        failure_over_time = []
        round_indices = []
        # Initialize temporary tracking
        temp_success_freq = [dict() for _ in range(config.num_items)]
        temp_failure_freq = [dict() for _ in range(config.num_items)]

    try:
        for round_idx in range(config.time_horizon):
            # Stop if budget is exhausted
            if agent.remaining_budget <= 0:
                raise BudgetDepletedException()

            # Agent selects price indices
            price_indices = agent.select_prices()
            
            # Map indices to actual prices
            selected_prices = [
                config.price_range[price_idx] for price_idx in price_indices
            ]
            
            # Get purchase decisions from environment
            purchase_decisions = env.get_purchase_decisions(round_idx, selected_prices)
            
            # Calculate rewards and costs
            item_rewards = np.array([
                price if purchased else 0 
                for price, purchased in zip(selected_prices, purchase_decisions)
            ])
            
            # Update price frequencies (total attempts)
            _update_price_frequencies(price_frequencies, selected_prices)
            
            # Update success/failure frequencies
            _update_success_failure_frequencies(success_frequencies, failure_frequencies, 
                                              selected_prices, purchase_decisions)
            
            # Update time series data if tracking
            if track_time_series:
                _update_success_failure_frequencies(temp_success_freq, temp_failure_freq,
                                                  selected_prices, purchase_decisions)
                # Save snapshot every 50 rounds for animation
                if (round_idx + 1) % 50 == 0 or round_idx == config.time_horizon - 1:
                    success_over_time.append([dict(freq) for freq in temp_success_freq])
                    failure_over_time.append([dict(freq) for freq in temp_failure_freq])
                    round_indices.append(round_idx + 1)
            
            # Each sale consumes 1 budget unit
            item_costs = np.array(purchase_decisions, dtype=int)
            
            total_round_reward = item_rewards.sum()
            agent.update_statistics(price_indices, item_rewards, item_costs)
            agent_rewards.append(total_round_reward)
            
    except BudgetDepletedException:
        # Complete remaining rounds with zero rewards
        remaining_rounds = config.time_horizon - len(agent_rewards)
        if remaining_rounds > 0:
            agent_rewards.extend([0] * remaining_rounds)
    finally:
        budget_depleted_round = agent.current_round
        
        # Create time series data if tracking was enabled
        if track_time_series:
            time_series_data = TimeSeriesData(
                success_freq_over_time=success_over_time,
                failure_freq_over_time=failure_over_time,
                round_indices=round_indices
            )

    return SingleSimulationResult(
        baseline_price_per_item=baseline_prices,
        arm_freq_per_item=price_frequencies,
        arm_success_freq_per_item=success_frequencies,
        arm_failure_freq_per_item=failure_frequencies,
        agent_reward_sums=np.array(agent_rewards),
        baseline_rewards=baseline_rewards,
        budget_depleted_round=budget_depleted_round,
        time_series_data=time_series_data,
    )


def _update_price_frequencies(price_frequencies: List[Dict], selected_prices: List[float]) -> None:
    """
    Update frequencies of selected prices.
    
    Args:
        price_frequencies: List of dictionaries to track frequencies
        selected_prices: Prices selected in this round
    """
    for item_idx, price in enumerate(selected_prices):
        if price not in price_frequencies[item_idx]:
            price_frequencies[item_idx][price] = 0
        price_frequencies[item_idx][price] += 1


def _update_success_failure_frequencies(success_frequencies: List[Dict], 
                                      failure_frequencies: List[Dict],
                                      selected_prices: List[float], 
                                      purchase_decisions: List[bool]) -> None:
    """
    Update frequencies of successful and failed purchases for each price.
    
    Args:
        success_frequencies: List of dictionaries to track successful purchases
        failure_frequencies: List of dictionaries to track failed purchases
        selected_prices: Prices selected in this round
        purchase_decisions: Whether each item was purchased
    """
    for item_idx, (price, purchased) in enumerate(zip(selected_prices, purchase_decisions)):
        if purchased:
            # Successful purchase
            if price not in success_frequencies[item_idx]:
                success_frequencies[item_idx][price] = 0
            success_frequencies[item_idx][price] += 1
        else:
            # Failed purchase
            if price not in failure_frequencies[item_idx]:
                failure_frequencies[item_idx][price] = 0
            failure_frequencies[item_idx][price] += 1


def run_multiple_simulations(
    agent_builder: Callable[[], CombinatorialUCBBidding],
    env_builder: Callable[[], Environment],
    config: ExperimentConfig,
) -> AggregatedSimulationResult:
    """
    Run multiple simulations and aggregate results.
    
    Args:
        agent_builder: Function to create new agent instances
        env_builder: Function to create new environments
        config: Experiment configuration
        
    Returns:
        Aggregated results from all simulations
    """
    # Initialize arrays to collect results
    cumulative_regrets = np.zeros((config.num_trials, config.time_horizon))
    total_baseline_rewards = np.zeros((config.num_trials, config.time_horizon))
    total_agent_rewards = np.zeros((config.num_trials, config.time_horizon))
    budget_depleted_rounds = np.zeros(config.num_trials, dtype=int)
    
    baseline_prices_per_item = np.zeros((config.num_trials, config.num_items))
    aggregated_price_frequencies = [dict() for _ in range(config.num_items)]
    aggregated_success_frequencies = [dict() for _ in range(config.num_items)]
    aggregated_failure_frequencies = [dict() for _ in range(config.num_items)]

    for trial_idx in range(config.num_trials):
        print(f"Running trial {trial_idx + 1}/{config.num_trials}")
        
        # Create new instances for each trial
        agent = agent_builder()
        env = env_builder()
        
        # Run simulation
        simulation_result = run_single_simulation(agent, env, config)

        # Calculate and store regret
        regret_per_round = simulation_result.baseline_rewards - simulation_result.agent_reward_sums
        cumulative_regrets[trial_idx] = np.cumsum(regret_per_round)
        
        # Store results
        total_baseline_rewards[trial_idx] = simulation_result.baseline_rewards
        total_agent_rewards[trial_idx] = simulation_result.agent_reward_sums
        budget_depleted_rounds[trial_idx] = simulation_result.budget_depleted_round
        baseline_prices_per_item[trial_idx] = np.array(simulation_result.baseline_price_per_item)
        
        # Aggregate price frequencies
        _aggregate_price_frequencies(aggregated_price_frequencies, 
                                   simulation_result.arm_freq_per_item, config.num_items)
        
        # Aggregate success/failure frequencies
        _aggregate_success_failure_frequencies(aggregated_success_frequencies, 
                                             simulation_result.arm_success_freq_per_item, config.num_items)
        _aggregate_success_failure_frequencies(aggregated_failure_frequencies, 
                                             simulation_result.arm_failure_freq_per_item, config.num_items)

    # Calculate mean frequencies
    _normalize_price_frequencies(aggregated_price_frequencies, config.num_trials, config.num_items)
    _normalize_success_failure_frequencies(aggregated_success_frequencies, config.num_trials, config.num_items)
    _normalize_success_failure_frequencies(aggregated_failure_frequencies, config.num_trials, config.num_items)

    return AggregatedSimulationResult(
        cumulative_regrets=cumulative_regrets,
        mean_arm_freq_per_item=aggregated_price_frequencies,
        mean_arm_success_freq_per_item=aggregated_success_frequencies,
        mean_arm_failure_freq_per_item=aggregated_failure_frequencies,
        total_baseline_rewards=total_baseline_rewards,
        total_agent_rewards=total_agent_rewards,
        budget_depleted_mean_round=budget_depleted_rounds.mean(),
        baseline_prices_per_item=baseline_prices_per_item,
    )


def _aggregate_price_frequencies(aggregated_frequencies: List[Dict], 
                               trial_frequencies: Collection[Dict[int, float]], 
                               num_items: int) -> None:
    """
    Aggregate price frequencies from a single trial.
    
    Args:
        aggregated_frequencies: Frequencies aggregated so far
        trial_frequencies: Frequencies from current trial
        num_items: Number of items
    """
    for item_idx in range(num_items):
        for price, frequency in trial_frequencies[item_idx].items():
            if price not in aggregated_frequencies[item_idx]:
                aggregated_frequencies[item_idx][price] = 0
            aggregated_frequencies[item_idx][price] += frequency


def _normalize_price_frequencies(aggregated_frequencies: List[Dict], 
                               num_trials: int, num_items: int) -> None:
    """
    Normalize aggregated frequencies by dividing by number of trials.
    
    Args:
        aggregated_frequencies: Aggregated frequencies to normalize
        num_trials: Total number of trials
        num_items: Number of items
    """
    for item_idx in range(num_items):
        for price in aggregated_frequencies[item_idx]:
            aggregated_frequencies[item_idx][price] /= num_trials


def _aggregate_success_failure_frequencies(aggregated_frequencies: List[Dict], 
                                         trial_frequencies: Collection[Dict[float, int]], 
                                         num_items: int) -> None:
    """
    Aggregate success/failure frequencies from a single trial.
    
    Args:
        aggregated_frequencies: Frequencies aggregated so far
        trial_frequencies: Frequencies from current trial
        num_items: Number of items
    """
    for item_idx in range(num_items):
        for price, frequency in trial_frequencies[item_idx].items():
            if price not in aggregated_frequencies[item_idx]:
                aggregated_frequencies[item_idx][price] = 0
            aggregated_frequencies[item_idx][price] += frequency


def _normalize_success_failure_frequencies(aggregated_frequencies: List[Dict], 
                                         num_trials: int, num_items: int) -> None:
    """
    Normalize aggregated success/failure frequencies by dividing by number of trials.
    
    Args:
        aggregated_frequencies: Aggregated frequencies to normalize
        num_trials: Total number of trials
        num_items: Number of items
    """
    for item_idx in range(num_items):
        for price in aggregated_frequencies[item_idx]:
            aggregated_frequencies[item_idx][price] /= num_trials


# Main simulation function with simplified interface
def simulate(
    agent_builder: Callable[[], CombinatorialUCBBidding],
    env_builder: Callable[[], Environment],
    config: ExperimentConfig,
) -> AggregatedSimulationResult:
    """
    Run multiple simulations with the given configuration.
    
    Args:
        agent_builder: Function to create new agent instances
        env_builder: Function to create new environments
        config: Complete experiment configuration
        
    Returns:
        Aggregated results from all simulations
    """
    return run_multiple_simulations(agent_builder, env_builder, config)


# Legacy compatibility functions (deprecated - use ExperimentConfig instead)
@dataclass
class SimulationConfig:
    """
    DEPRECATED: Use ExperimentConfig instead.
    Legacy configuration for backward compatibility only.
    """
    num_rounds: int
    prices_per_item: List[List[float]]
    total_budget: int
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        import warnings
        warnings.warn(
            "SimulationConfig is deprecated. Use ExperimentConfig instead.", 
            DeprecationWarning, 
            stacklevel=2
        )
        if self.num_rounds <= 0:
            raise ValueError("Number of rounds must be positive")
        if self.total_budget <= 0:
            raise ValueError("Total budget must be positive")
        if not self.prices_per_item or any(not prices for prices in self.prices_per_item):
            raise ValueError("Each item must have at least one available price")


def get_baseline(env: Environment, config) -> Tuple[Collection[float], np.ndarray]:
    """
    DEPRECATED: Use calculate_baseline_performance for compatibility.
    """
    import warnings
    warnings.warn(
        "get_baseline is deprecated. Use calculate_baseline_performance instead.", 
        DeprecationWarning, 
        stacklevel=2
    )
    if isinstance(config, ExperimentConfig):
        return calculate_baseline_performance(env, config)
    else:  # Legacy SimulationConfig
        # Convert to ExperimentConfig for processing
        temp_env_config = EnvironmentConfig(means=[20.0] * len(config.prices_per_item), 
                                          stds=[5.0] * len(config.prices_per_item))
        temp_experiment_config = ExperimentConfig(
            time_horizon=config.num_rounds,
            budget=config.total_budget,
            price_range=config.prices_per_item[0] if config.prices_per_item else [],
            num_trials=1,
            exploration_param=1.0,
            environment=temp_env_config
        )
        return calculate_baseline_performance(env, temp_experiment_config)


def run_simulation(agent: CombinatorialUCBBidding, env: Environment, 
                  config) -> SingleSimulationResult:
    """
    DEPRECATED: Use run_single_simulation for compatibility.
    """
    import warnings
    warnings.warn(
        "run_simulation is deprecated. Use run_single_simulation instead.", 
        DeprecationWarning, 
        stacklevel=2
    )
    if isinstance(config, ExperimentConfig):
        return run_single_simulation(agent, env, config)
    else:  # Legacy SimulationConfig
        # Convert to ExperimentConfig for processing
        temp_env_config = EnvironmentConfig(means=[20.0] * len(config.prices_per_item), 
                                          stds=[5.0] * len(config.prices_per_item))
        temp_experiment_config = ExperimentConfig(
            time_horizon=config.num_rounds,
            budget=config.total_budget,
            price_range=config.prices_per_item[0] if config.prices_per_item else [],
            num_trials=1,
            exploration_param=1.0,
            environment=temp_env_config
        )
        return run_single_simulation(agent, env, temp_experiment_config) 
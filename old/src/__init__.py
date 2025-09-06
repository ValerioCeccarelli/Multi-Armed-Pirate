"""
Multi-Armed Pirate: Combinatorial Pricing Simulation with UCB

This package implements a simulation system for the combinatorial pricing problem 
using Upper Confidence Bound (UCB) algorithms with budget constraints.

Main modules:
- agent: Contains the CombinatorialUCBBidding agent
- environment: Contains the Environment class for market simulation
- core: Main simulation functions and data structures
- plotting: Utilities for result visualization
"""

from .agent import CombinatorialUCBBidding
from .environment import Environment
from .core import (
    ExperimentConfig,
    EnvironmentConfig,
    SingleSimulationResult, 
    AggregatedSimulationResult,
    BudgetDepletedException,
    run_single_simulation,
    run_multiple_simulations,
    calculate_baseline_performance,
    simulate,
    # Legacy compatibility (deprecated)
    SimulationConfig,
    run_simulation,
    get_baseline
)
from .plotting import (
    PlottingConfig,
    generate_comprehensive_plots,
    plot_cumulative_regret,
    plot_cumulative_rewards,
    plot_price_frequency_histograms,
    print_simulation_summary,
    # Aliases for compatibility
    plot_statistics,
    show_regret,
    show_cumulative_rewards,
    show_histogram
)

__version__ = "1.0.0"
__author__ = "Multi-Armed Pirate Team"

__all__ = [
    # Main classes
    "CombinatorialUCBBidding",
    "Environment",
    
    # Main configurations
    "ExperimentConfig", 
    "EnvironmentConfig",
    "PlottingConfig",
    
    # Results
    "SingleSimulationResult",
    "AggregatedSimulationResult",
    "BudgetDepletedException",
    
    # Main simulation functions
    "simulate",
    "run_single_simulation",
    "run_multiple_simulations", 
    "calculate_baseline_performance",
    
    # Plotting functions
    "generate_comprehensive_plots",
    "plot_cumulative_regret",
    "plot_cumulative_rewards",
    "plot_price_frequency_histograms",
    "print_simulation_summary",
    "plot_statistics",
    "show_regret",
    "show_cumulative_rewards",
    "show_histogram",
    
    # Legacy compatibility (deprecated)
    "SimulationConfig",
    "run_simulation",
    "get_baseline",
]


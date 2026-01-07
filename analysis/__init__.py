"""Analysis module for POCII project."""

from .classifier_analysis import (
    generate_all_plots,
    analyze_tradeoff,
    print_analysis_summary,
    plot_f1_heatmap,
    plot_execution_time,
    plot_tradeoff_scatter,
    plot_f1_boxplot,
    plot_performance_by_dataset
)

try:
    from .analyze_optuna_results import OptunaResultsAnalyzer
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    OptunaResultsAnalyzer = None

__all__ = [
    'generate_all_plots',
    'analyze_tradeoff',
    'print_analysis_summary',
    'plot_f1_heatmap',
    'plot_execution_time',
    'plot_tradeoff_scatter',
    'plot_f1_boxplot',
    'plot_performance_by_dataset',
    'OptunaResultsAnalyzer'
]

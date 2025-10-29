"""Evaluation metrics"""

from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
from scipy import stats


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    environment: str
    num_samples: int
    accuracy: float
    mean_score: float
    std_score: float
    median_score: float
    confidence_interval: tuple
    success_rate: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "environment": self.environment,
            "num_samples": self.num_samples,
            "accuracy": self.accuracy,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "median_score": self.median_score,
            "confidence_interval": {
                "lower": self.confidence_interval[0],
                "upper": self.confidence_interval[1]
            },
            "success_rate": self.success_rate,
            "metadata": self.metadata
        }


def compute_confidence_interval(
    scores: List[float],
    confidence_level: float = 0.80
) -> tuple:
    """
    Compute confidence interval using Beta distribution (Bayesian approach)

    Args:
        scores: List of binary scores (0.0 or 1.0)
        confidence_level: Confidence level (default: 0.80)

    Returns:
        (lower_bound, upper_bound) tuple
    """
    if not scores:
        return (0.0, 0.0)

    # Count successes and failures
    successes = sum(scores)
    trials = len(scores)

    # Use Jeffrey's prior (alpha=0.5, beta=0.5)
    alpha = 0.5 + successes
    beta = 0.5 + (trials - successes)

    # Compute confidence interval
    lower_percentile = (1 - confidence_level) / 2
    upper_percentile = 1 - lower_percentile

    lower_bound = stats.beta.ppf(lower_percentile, alpha, beta)
    upper_bound = stats.beta.ppf(upper_percentile, alpha, beta)

    return (lower_bound, upper_bound)


def compute_metrics(
    environment: str,
    scores: List[float],
    metadata: Dict[str, Any] = None
) -> EvaluationMetrics:
    """
    Compute evaluation metrics for a given environment

    Args:
        environment: Environment name
        scores: List of scores
        metadata: Additional metadata

    Returns:
        EvaluationMetrics object
    """
    if not scores:
        return EvaluationMetrics(
            environment=environment,
            num_samples=0,
            accuracy=0.0,
            mean_score=0.0,
            std_score=0.0,
            median_score=0.0,
            confidence_interval=(0.0, 0.0),
            success_rate=0.0,
            metadata=metadata or {}
        )

    scores = np.array(scores)

    # Compute basic statistics
    num_samples = len(scores)
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    median_score = float(np.median(scores))

    # Compute accuracy (for binary tasks)
    accuracy = float(np.mean(scores > 0.5))

    # Compute success rate (proportion of non-zero scores)
    success_rate = float(np.mean(scores > 0.0))

    # Compute confidence interval
    confidence_interval = compute_confidence_interval(scores.tolist())

    return EvaluationMetrics(
        environment=environment,
        num_samples=num_samples,
        accuracy=accuracy,
        mean_score=mean_score,
        std_score=std_score,
        median_score=median_score,
        confidence_interval=confidence_interval,
        success_rate=success_rate,
        metadata=metadata or {}
    )


def aggregate_metrics(metrics_list: List[EvaluationMetrics]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple environments

    Args:
        metrics_list: List of EvaluationMetrics

    Returns:
        Dictionary with aggregated metrics
    """
    if not metrics_list:
        return {}

    total_samples = sum(m.num_samples for m in metrics_list)
    weighted_accuracy = sum(m.accuracy * m.num_samples for m in metrics_list) / total_samples if total_samples > 0 else 0.0
    weighted_mean_score = sum(m.mean_score * m.num_samples for m in metrics_list) / total_samples if total_samples > 0 else 0.0

    # Compute overall success rate
    overall_success_rate = np.mean([m.success_rate for m in metrics_list])

    # Find best and worst environments
    best_env = max(metrics_list, key=lambda m: m.mean_score)
    worst_env = min(metrics_list, key=lambda m: m.mean_score)

    return {
        "total_samples": total_samples,
        "num_environments": len(metrics_list),
        "weighted_accuracy": weighted_accuracy,
        "weighted_mean_score": weighted_mean_score,
        "overall_success_rate": overall_success_rate,
        "best_environment": {
            "name": best_env.environment,
            "score": best_env.mean_score
        },
        "worst_environment": {
            "name": worst_env.environment,
            "score": worst_env.mean_score
        },
        "per_environment": {
            m.environment: m.to_dict() for m in metrics_list
        }
    }


def check_pareto_dominance(
    scores_a: Dict[str, float],
    scores_b: Dict[str, float]
) -> int:
    """
    Check if scores_a Pareto dominates scores_b

    Args:
        scores_a: Dictionary of environment -> score for model A
        scores_b: Dictionary of environment -> score for model B

    Returns:
        1 if A dominates B, -1 if B dominates A, 0 if neither
    """
    a_better = False
    b_better = False

    for env in scores_a:
        if env not in scores_b:
            continue

        if scores_a[env] > scores_b[env]:
            a_better = True
        elif scores_a[env] < scores_b[env]:
            b_better = True

    if a_better and not b_better:
        return 1  # A dominates B
    elif b_better and not a_better:
        return -1  # B dominates A
    else:
        return 0  # Neither dominates


def compute_pareto_score(
    model_scores: Dict[str, float],
    baseline_scores: Dict[str, float]
) -> float:
    """
    Compute Pareto score relative to baseline

    Args:
        model_scores: Dictionary of environment -> score for model
        baseline_scores: Dictionary of environment -> score for baseline

    Returns:
        Pareto score (percentage of environments where model wins or ties)
    """
    wins = 0
    ties = 0
    total = 0

    for env in model_scores:
        if env not in baseline_scores:
            continue

        total += 1
        if model_scores[env] > baseline_scores[env]:
            wins += 1
        elif model_scores[env] == baseline_scores[env]:
            ties += 1

    if total == 0:
        return 0.0

    return (wins + 0.5 * ties) / total

from typing import NamedTuple
import numpy as np

from src.experiments.scoring import aggregate_scores
from src.experiments.methods import ExperimentsResult

class MetricSummary(NamedTuple):
  values: np.ndarray
  median: float
  mean: float
  std: float
  min: float
  max: float

class ScoreSummary(NamedTuple):
  accuracy: MetricSummary
  recall: MetricSummary
  precision: MetricSummary
  confusion: np.ndarray

def summarize_metric(scores: list[float]) -> MetricSummary: return MetricSummary(
  np.array(scores),
  np.median(scores),
  np.mean(scores),
  np.std(scores),
  np.min(scores),
  np.max(scores)
)

ExperimentsSummary = dict[str, ScoreSummary]

def summarize_results(result: ExperimentsResult) -> ExperimentsSummary:
  return {
    key: ScoreSummary(
      summarize_metric(aggregate.accuracy),
      summarize_metric(aggregate.recall),
      summarize_metric(aggregate.precision),
      aggregate.confusion_matrix
    )
    for key, scores in result.items()
    for aggregate in [aggregate_scores(scores)]
  }

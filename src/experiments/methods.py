from joblib import Parallel, delayed
from typing import NamedTuple, Callable, Mapping
from pandas import DataFrame
from src.experiments.models import Model
from src.experiments.scoring import score_model, ScoreResult
import functools as ft
import itertools as it

class ExperimentResult(NamedTuple):
  name: str
  scores: list[ScoreResult]

def run_experiment(
    dataset: DataFrame,
    dataset_label: str,
    create_model: Callable[[], Model],
    model_label: str,
    run_count: int,
    index: int,
    total: int
) -> ExperimentResult:
  print(f"Running dataset and model {index + 1}/{total}: {dataset_label} {model_label}")

  return ExperimentResult(
    f"{dataset_label}:{model_label}",
    [score_model(create_model(), dataset) for _ in range(run_count)]
  )

ExperimentsResult = dict[str, list[ScoreResult]]

def join_results(accumulator: ExperimentsResult, result: ExperimentResult) -> ExperimentsResult:
  accumulator[result.name] = result.scores
  return accumulator


def run_experiments(
    datasets: Mapping[str, DataFrame],
    model_descriptors: Mapping[str, Callable[[], Model]],
    run_count: int
) -> ExperimentsResult:
  run = delayed(run_experiment)
  total = len(datasets) * len(model_descriptors)

  return ft.reduce(
    join_results,
    Parallel(n_jobs=-1)(
      run(dataset, dataset_label, create_model, model_label, run_count, index, total)
      for index, ((dataset_label, dataset), (model_label, create_model)) in
      enumerate(it.product(datasets.items(), model_descriptors.items()))
    ),
    {}
  )

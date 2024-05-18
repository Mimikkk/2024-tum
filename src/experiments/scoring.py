from typing import NamedTuple

import numpy as np
from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score

from sklearn.model_selection import train_test_split

from src.experiments.models import Model

class SplitResult(NamedTuple):
  X_train: DataFrame
  X_test: DataFrame
  y_train: Series
  y_test: Series

def split_dataset(dataset: DataFrame) -> SplitResult:
  X = dataset.drop(columns=['target'])
  y = dataset['target']

  return SplitResult(*train_test_split(X, y, test_size=0.2))

class ScoreResult(NamedTuple):
  accuracy: float
  recall: float
  precision: float
  confusion_matrix: np.ndarray

def score_model(model: Model, frame: DataFrame) -> ScoreResult:
  x_train, x_test, y_train, y_test = split_dataset(frame)

  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)

  return ScoreResult(
    accuracy_score(y_test, y_pred),
    recall_score(y_test, y_pred, average='micro'),
    precision_score(y_test, y_pred, average='micro'),
    confusion_matrix(y_test, y_pred, normalize='true')
  )

class ScoreAggregate(NamedTuple):
  accuracy: list[float]
  recall: list[float]
  precision: list[float]
  confusion_matrix: np.ndarray

def aggregate_scores(results: list[ScoreResult]) -> ScoreAggregate:
  return ScoreAggregate(
    [result.accuracy for result in results],
    [result.recall for result in results],
    [result.precision for result in results],
    np.mean([result.confusion_matrix for result in results], axis=0)
  )

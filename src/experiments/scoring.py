from typing import NamedTuple

import numpy as np
from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score

from src.experiments.models import Model

class SplitResult(NamedTuple):
  X_train: DataFrame
  X_test: DataFrame
  y_train: Series
  y_test: Series

def split_dataset(dataset: tuple[DataFrame, DataFrame]):
  train_dataset, test_dataset = dataset

  total = len(train_dataset)
  train_indices = np.random.choice(train_dataset.index, int(total * 0.8), replace=False)
  np.random.shuffle(train_indices)

  test_indices = list(train_dataset.index.difference(train_indices))
  np.random.shuffle(test_indices)

  train = train_dataset.loc[train_indices]
  test = train_dataset.loc[test_indices]

  x_train = train.drop(columns=['target'])
  x_test = test.drop(columns=['target'])
  y_train = train['target']
  y_test = test['target']

  return SplitResult(x_train, x_test, y_train, y_test)

class ScoreResult(NamedTuple):
  accuracy: float
  recall: float
  precision: float
  confusion_matrix: np.ndarray

def score_model(model: Model, dataset: tuple[DataFrame, DataFrame]) -> ScoreResult:
  x_train, x_test, y_train, y_test = split_dataset(dataset)

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

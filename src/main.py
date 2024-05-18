from collections import defaultdict
from typing import NamedTuple, Protocol, Callable, Mapping

from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.datasets import ModelIris, ModelWine, DatasetWine, DatasetIris, DatasetNoise
import numpy as np

from src.memo import memo

# 2. Implementacja algorytmów uczenia maszynowego
#     - Wykorzystano algorytmy: SVM, KNN, Random Forest, oraz prostą sieć neuronową MLP
#     - Wykorzystano gotowe implementacje z biblioteki sklearn oraz keras
#     - Zaimplementowano funkcję, która zwraca trafność klasyfikacji dla zbioru danych

class SplitResult(NamedTuple):
  X_train: DataFrame
  X_test: DataFrame
  y_train: Series
  y_test: Series

from sklearn.model_selection import train_test_split

def split_dataset(dataset: DataFrame) -> SplitResult:
  X = dataset.drop(columns=['target'])
  y = dataset['target']

  return SplitResult(*train_test_split(X, y, test_size=0.2))

class Model(Protocol):
  def fit(self, X: DataFrame, y: Series) -> None:
    ...

  def predict(self, X: DataFrame) -> Series:
    ...

def score_model(create_model: Callable[[], Model], frame: DataFrame) -> float:
  model = create_model()
  x_train, x_test, y_train, y_test = split_dataset(frame)

  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)

  return accuracy_score(y_test, y_pred)

import itertools as it

@memo(type='file', hashstr='base_datasets', verbose=True)
def load_datasets() -> tuple[DatasetIris, DatasetWine]:
  iris = ModelIris.load()
  wine = ModelWine.load()
  return iris, wine

@memo(type='file', hashstr='noise_datasets', verbose=True)
def create_noisy_datasets(iris: DatasetIris, wine: DatasetWine) -> Mapping[str, DataFrame]:
  datasets = {
    "iris-base": iris,
    "wine-base": wine,
  }

  for (
      dataset,
      use_noise_input,
      random_input_type,
      random_input_scale,
      random_input_count,
      use_noise_output,
      random_output_scale) \
      in it.product(
    (iris, wine),
    (False, True),
    ('static', 'random', 'corr-static', 'corr-random'),
    (0.1, 0.25, 0.5, 1.0),
    range(1, 6),
    (False, True),
    (0.1, 0.25, 0.5, 1.0),
  ):
    dataset_label = 'iris' if dataset is iris else 'wine'

    if use_noise_input:
      match random_input_type:
        case 'static':
          dataset_label += f'_i-noise-{random_input_type}-{random_input_count}-({random_input_scale})'
        case 'random':
          dataset_label += f'_i-noise-{random_input_type}-{random_input_count}-({-random_input_scale}, {random_input_scale})'
        case 'corr-static':
          dataset_label += f'_i-noise-{random_input_type}-{random_input_count}'
        case 'corr-random':
          dataset_label += f'_i-noise-{random_input_type}-{random_input_count}'

    if use_noise_output:
      dataset_label += f'_o-noise-{random_output_scale}'

    if dataset_label in datasets: continue

    noise = DatasetNoise(dataset)
    if use_noise_input:
      match random_input_type:
        case 'static':
          noise.add_static_noises('noise', random_input_count, random_input_scale)
        case 'random':
          noise.add_random_noises('noise', random_input_count, (-random_input_scale, random_input_scale))
        case 'corr-static':
          column = dataset.columns[np.random.randint(0, len(dataset.columns))]

          noise.add_static_correlated_noises('noise', column, random_input_count)
        case 'corr-random':
          column = dataset.columns[np.random.randint(0, len(dataset.columns))]

          noise.add_random_correlated_noises('noise', column, random_input_count)

    if use_noise_output:
      noise.shuffle('target', random_output_scale)

    datasets[dataset_label] = noise.build()
  return datasets

def create_model_descriptors() -> Mapping[str, Callable[[], Model]]:
  return {
    "SVC": lambda: SVC(),
    "KNN2": lambda: KNeighborsClassifier(n_neighbors=2),
    "KNN3": lambda: KNeighborsClassifier(n_neighbors=3),
    "KNN4": lambda: KNeighborsClassifier(n_neighbors=4),
    "RandomForest": lambda: RandomForestClassifier(max_depth=8),
    "MLP": lambda: MLPClassifier(max_iter=100_000),
  }

@memo(type='file', hashstr='experiments', verbose=True)
def run_experiments(
    datasets: Mapping[str, DataFrame],
    model_descriptors: Mapping[str, Callable[[], Model]],
    run_count: int
) -> Mapping[
  str, list[float]]:
  score_map: Mapping[str, list[float]] = defaultdict(list)

  for dataset_label, dataset in datasets.items():
    print(f"Running dataset: {dataset_label}")
    for model_label, create_model in model_descriptors.items():
      print(f"- Running model: {model_label}")
      score_map[f'{dataset_label}:{model_label}'] = [
        score_model(create_model, dataset)
        for _ in range(run_count)
      ]

  return score_map

@memo(type='file', hashstr='results', verbose=True)
def summarize_results(score_map: Mapping[str, list[float]]) -> Mapping[str, Mapping[str, float]]:
  results = {}
  for key, scores in score_map.items():
    metrics = {'mean': np.mean, 'median': np.median, 'std': np.std, 'min': np.min, 'max': np.max}
    results[key] = {metric: func(scores) for metric, func in metrics.items()}
  return results

def main():
  run_count = 1
  iris, wine = load_datasets()
  datasets = create_noisy_datasets(iris, wine)
  model_descriptors = create_model_descriptors()
  results = run_experiments(datasets, model_descriptors, run_count=run_count)

  import pickle as pkl

  with open('results.pkl', 'wb') as file:
    pkl.dump(results, file)

if __name__ == '__main__':
  main()

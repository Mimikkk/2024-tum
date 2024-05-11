from typing import NamedTuple, Protocol, Callable

from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.datasets import ModelIris, ModelWine, DatasetNoise

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

def main():
  iris = ModelIris.load()
  wine = ModelWine.load()

  datasets = {
    "iris-base": iris,
    "iris-static_noise1": DatasetNoise(iris).add_static_noise('noise', 0.1).build(),
    "iris-static_noise5": DatasetNoise(iris).add_static_noises('noise', 5, 0.1).build(),
    "wine-base": wine,
  }

  model_factories = {
    "SVC": lambda: SVC(),
    "KNN3": lambda: KNeighborsClassifier(n_neighbors=3),
    "RandomForest": lambda: RandomForestClassifier(max_depth=8),
    "MLP": lambda: MLPClassifier(max_iter=100_000),
  }

  for dataset_label, dataset in datasets.items():
    print(f"{dataset_label} dataset:")
    for model_label, factory in model_factories.items():
      score = score_model(factory, dataset)
      print(f"\t{model_label}: {score * 100:.4f}%")
    print()

if __name__ == '__main__':
  main()

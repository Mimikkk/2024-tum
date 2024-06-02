from typing import Protocol, Callable, Mapping

from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

class Model(Protocol):
  def fit(self, X: DataFrame, y: Series) -> None:
    ...

  def predict(self, X: DataFrame) -> Series:
    ...

def create_model_descriptors() -> Mapping[str, Callable[[], Model]]: return {
  "SVC": lambda: SVC(),
  "KNN2": lambda: KNeighborsClassifier(n_neighbors=2),
  "KNN3": lambda: KNeighborsClassifier(n_neighbors=3),
  "KNN4": lambda: KNeighborsClassifier(n_neighbors=4),
  "RandomForest": lambda: RandomForestClassifier(max_depth=8),
  "MLP": lambda: MLPClassifier(max_iter=100_000),
}

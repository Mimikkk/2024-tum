import numpy as np
from pandas import DataFrame, Series

# Implementacja sposobów dodawania szumu
# Dodawanie szumu nowych cech wejściowych z czystym szumem
# Dodawanie szumu nowych cech wejściowych ze skorelowanym szumem
# Dodawanie szumu do istniejących cech wejściowych z czystym szumem
# Dodawanie szumu do istniejących cech wejściowych ze skorelowanym szumem
# Dodawanie szumu na wyjściu (losowe zmiany etykiet klas w części obiektów)

def norm_column(column: Series) -> Series:
  return (column - column.min()) / (column.max() - column.min())

def scale_random(count: int, scale: tuple[float, float]) -> np.ndarray:
  min_scale, max_scale = scale
  return min_scale + (max_scale - min_scale) * np.random.rand(count)

def create_static_noise(column: Series, scale: float | Series | np.ndarray) -> Series:
  return (column.max() - column.min()) * scale

def create_random_noise(column: Series, scale: tuple[float, float]) -> Series:
  return create_static_noise(column, scale_random(column.size, scale))

def create_static_correlated_noise(column: Series, correlated_column: Series) -> Series:
  return create_static_noise(column, norm_column(correlated_column))

def create_random_correlated_noise(column: Series, correlated_column: Series) -> Series:
  return create_static_correlated_noise(column, correlated_column * np.random.rand(column.size))

class DatasetNoise:
  def __init__(self, frame: DataFrame):
    self.frame = frame.copy(deep=True)

  def add_noise(self, column_id: str, noise: Series | float | np.ndarray) -> 'DatasetNoise':
    self.frame[column_id] += noise
    return self

  def add_empty(self, column_id: str) -> 'DatasetNoise':
    return self.add_noise(column_id, 0.0)

  def add_static_noise(self, column_id: str, scale: float) -> 'DatasetNoise':
    if column_id not in self.frame: self.add_empty(column_id)
    return self.add_noise(column_id, create_static_noise(self.frame[column_id], scale))

  def add_random_noise(self, column_id: str, scale: tuple[float, float]) -> 'DatasetNoise':
    if column_id not in self.frame: self.add_empty(column_id)
    return self.add_noise(column_id, create_random_noise(self.frame[column_id], scale))

  def add_static_correlated_noise(self, column_id: str, correlated_column_id: str) -> 'DatasetNoise':
    if column_id not in self.frame: self.add_empty(column_id)
    column_a = self.frame[column_id]
    column_b = self.frame[correlated_column_id]
    return self.add_noise(column_id, create_static_correlated_noise(column_a, column_b))

  def add_random_correlated_noise(self, column_id: str, correlated_column_id: str) -> 'DatasetNoise':
    if column_id not in self.frame: self.add_empty(column_id)
    column_a = self.frame[column_id]
    column_b = self.frame[correlated_column_id]
    return self.add_noise(column_id, create_random_correlated_noise(column_a, column_b))

  def shuffle(self, column_id: str, percentage: float) -> 'DatasetNoise':
    col = self.frame[column_id]
    indices = col.sample(frac=percentage).index
    col.loc[indices] = col.sample(frac=1).values
    return self

  def build(self) -> DataFrame:
    return self.frame

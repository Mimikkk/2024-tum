from typing import Mapping
import itertools as it
import numpy as np
from pandas import DataFrame

from src.datasets.dataset_noise import DatasetNoise
from src.datasets.dataset_iris import ModelIris, DatasetIris
from src.datasets.dataset_wine import ModelWine, DatasetWine

# @memo(type='file', hashstr='base_datasets', verbose=True)
def load_datasets() -> tuple[DatasetIris, DatasetWine]:
  iris = ModelIris.load()
  wine = ModelWine.load()
  return iris, wine

# @memo(type='file', hashstr='noise_datasets', verbose=True)
def create_noisy_datasets(iris: DatasetIris, wine: DatasetWine) -> Mapping[str, DataFrame]:
  datasets = {}
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
    ('static', 'random'),
    (0.1,),
    range(1, 5),
    (False,),
    (0.05,),
    # (iris, wine),
    # (False, True),
    # ('static', 'random', 'corr-static', 'corr-random'),
    # (0.1, 0.25, 0.5, 1.0),
    # range(1, 6),
    # (False, True),
    # (0.05, 0.1, 0.25, 0.5, 0.75, 1.0),
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

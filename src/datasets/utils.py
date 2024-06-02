from typing import Mapping
import itertools as it
import numpy as np
from pandas import DataFrame

from src.datasets.dataset_noise import DatasetNoise
from src.datasets.dataset_iris import ModelIris, DatasetIris
from src.datasets.dataset_wine import ModelWine, DatasetWine

def pick_random_column_id(dataset: DataFrame) -> str:
  return dataset.columns[np.random.randint(0, len(dataset.columns) - 1)]

def pick_two_random_column_ids(dataset: DataFrame) -> tuple[str, str]:
  return np.random.choice(dataset.columns[:-1], 2, replace=False)

def load_datasets() -> tuple[DatasetIris, DatasetWine]:
  iris = ModelIris.load()
  wine = ModelWine.load()
  return iris, wine

def create_noisy_datasets(iris: DatasetIris, wine: DatasetWine) -> Mapping[str, tuple[DataFrame, DataFrame]]:
  datasets = {}

  dataset: DataFrame
  for (
      dataset,
      use_create_noise_input,
      random_input_type,
      random_input_scale,
      random_input_count,
      use_modify_noise_input,
      modified_input_type,
      modified_input_scale,
      use_noise_output,
      random_output_scale
  ) \
      in it.product(
    (iris, wine),
    (False, True),
    ('static', 'random', 'corr-static', 'corr-random'),
    (0.1, 0.25, 0.75),
    range(1, 3),
    (False, True),
    ('static', 'random', 'corr-static', 'corr-random'),
    (0.1, 0.25, 0.75),
    (False, True),
    (0.1, 0.25, 0.75),
  ):
    dataset_label = 'iris' if dataset is iris else 'wine'

    if use_create_noise_input:
      match random_input_type:
        case 'static':
          dataset_label += f'_io-noise-{random_input_type}-{random_input_count}-({random_input_scale})'
        case 'random':
          dataset_label += f'_io-noise-{random_input_type}-{random_input_count}-({-random_input_scale}, {random_input_scale})'
        case 'corr-static':
          dataset_label += f'_io-noise-{random_input_type}-{random_input_count}'
        case 'corr-random':
          dataset_label += f'_io-noise-{random_input_type}-{random_input_count}'

    if use_modify_noise_input:
      match modified_input_type:
        case 'static':
          dataset_label += f'_im-noise-{modified_input_type}'
        case 'random':
          dataset_label += f'_im-noise-{modified_input_type}-({-modified_input_scale}, {modified_input_scale})'
        case 'corr-static':
          dataset_label += f'_im-noise-{modified_input_type}'
        case 'corr-random':
          dataset_label += f'_im-noise-{modified_input_type}'

    if use_noise_output:
      dataset_label += f'_o-noise-{random_output_scale}'

    if dataset_label in datasets: continue

    noise = DatasetNoise(dataset)
    if use_create_noise_input:
      match random_input_type:
        case 'static':
          noise.add_static_noises('noise', random_input_count, random_input_scale)
        case 'random':
          scale = (-random_input_scale, random_input_scale)
          noise.add_random_noises('noise', random_input_count, scale)
        case 'corr-static':
          column_id = pick_random_column_id(dataset)
          noise.add_static_correlated_noises('noise', column_id, random_input_count)
        case 'corr-random':
          scale = (-random_input_scale, random_input_scale)
          column_id = pick_random_column_id(dataset)
          noise.add_random_correlated_noises('noise', column_id, random_input_count, scale)

    if use_modify_noise_input:
      match modified_input_type:
        case 'static':
          column_id = pick_random_column_id(dataset)

          noise.add_static_noise(column_id, modified_input_scale)
        case 'random':
          column_id = pick_random_column_id(dataset)
          scale = (-modified_input_scale, modified_input_scale)

          noise.add_random_noise(column_id, scale)
        case 'corr-static':
          first_id, second_id = pick_two_random_column_ids(dataset)

          noise.add_static_correlated_noise(first_id, second_id)
        case 'corr-random':
          first_id, second_id = pick_two_random_column_ids(dataset)

          scale = (-modified_input_scale, modified_input_scale)
          noise.add_random_correlated_noise(first_id, second_id, scale)

    if use_noise_output:
      noise.shuffle('target', random_output_scale)

    noisy = noise.build()
    clean = dataset.copy(deep=True)

    datasets[dataset_label] = (clean, noisy)
  return datasets

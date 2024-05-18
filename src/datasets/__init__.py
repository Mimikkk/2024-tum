from src.datasets.dataset_noise import DatasetNoise
from src.datasets.dataset_iris import ModelIris, DatasetIris
from src.datasets.dataset_wine import ModelWine, DatasetWine
from src.datasets.utils import load_datasets, create_noisy_datasets

Dataset = DatasetIris | DatasetWine

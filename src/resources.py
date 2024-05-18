import pathlib
import pickle

from src.experiments.methods import ExperimentsResult

ResourcesDirectory = pathlib.Path('../resources')
ResultsDirectory = ResourcesDirectory / 'results'

def save_result(result: ExperimentsResult) -> None:
  with open(ResultsDirectory / 'results.pkl', 'wb') as file:
    pickle.dump(result, file)

def load_result() -> ExperimentsResult:
  with open(ResultsDirectory / 'results.pkl', 'rb') as file:
    result = pickle.load(file)
  return result

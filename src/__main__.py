from analysis import visualize_correlation, visualize_metrics
from experiments import summarize_results, run_experiments, create_model_descriptors
from src.datasets.utils import load_datasets, create_noisy_datasets

def main():
  run_count = 100
  iris, wine = load_datasets()
  datasets = create_noisy_datasets(iris, wine)
  model_descriptors = create_model_descriptors()
  result = run_experiments(datasets, model_descriptors, run_count)
  summary = summarize_results(result)

  visualize_metrics(summary)
  visualize_correlation(summary)

if __name__ == '__main__':
  main()

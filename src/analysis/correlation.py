from typing import NamedTuple

from src.experiments import ExperimentsSummary
from scipy.stats import pearsonr, ttest_ind
from statsmodels.stats.power import TTestIndPower

class ExperimentsLabels(NamedTuple):
  experiments: list[str]
  datasets: list[str]
  models: list[str]
  noises: list[str]

def read_labels(summary: ExperimentsSummary) -> ExperimentsLabels:
  model_labels = set()
  dataset_labels = set()
  noise_labels = set()
  experiment_labels = sorted(summary.keys())

  for label in experiment_labels:
    dataset_label, model_label = label.split(':')

    model_labels.add(model_label)
    match dataset_label.split('_', 1):
      case [dataset_label, noise_label]:
        dataset_labels.add(dataset_label)
        noise_labels.add(noise_label)
      case [dataset_label]:
        dataset_labels.add(dataset_label)

  model_labels = sorted(model_labels)
  dataset_labels = sorted(dataset_labels)
  noise_labels = sorted(noise_labels)

  return ExperimentsLabels(experiment_labels, dataset_labels, model_labels, noise_labels)

def visualize_correlation(summary: ExperimentsSummary):
  metrics = ['accuracy', 'recall', 'precision']

  labels = read_labels(summary)

  for metric in metrics:
    for model_label in labels.models:
      for dataset_label in labels.datasets:
        base_model = summary[f'{dataset_label}:{model_label}']
        for noise_label in labels.noises:
          noise_model = summary[f'{dataset_label}_{noise_label}:{model_label}']

          base_metric = getattr(base_model, metric).values
          noise_metric = getattr(noise_model, metric).values
          correlation, p_value = pearsonr(base_metric, noise_metric)

          experiment_label = f'{dataset_label}_{noise_label}:{model_label}'
          print(f"Correlation:{experiment_label}: {correlation}, p-value: {p_value}")
          is_significant = p_value < 0.05

          print(f'Correlation p-value: {p_value}')
          if is_significant:
            print(f"- Significant correlation: {experiment_label}")
          else:
            print(f"- No significant correlation: {experiment_label}")

          t_test, _ = ttest_ind(base_metric, noise_metric)
          degree_of_freedom = len(base_metric) + len(noise_metric) - 2
          power = TTestIndPower().solve_power(abs(t_test), degree_of_freedom, 0.05, alternative='two-sided')
          print(f"- Statistical power: {power}")

          if power > 0.8:
            print(f"- Sufficient statistical power: {experiment_label}")
          else:
            print(f"- Insufficient statistical power: {experiment_label}")
          print()

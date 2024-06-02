from src.experiments import ExperimentsSummary
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_metrics(summary: ExperimentsSummary):
  # metrics = ['accuracy', 'recall', 'precision']
  metrics = ['accuracy']
  models = list(summary.keys())

  has_too_many_models = len(models) > 8

  for metric in metrics:
    plt.figure(figsize=(16, 8))
    means = [getattr(summary[model], metric).mean for model in models]
    plt.bar(models, means, color='skyblue')
    plt.title(f'Mean "{metric}" across different models')
    plt.ylabel(metric)
    plt.xticks(rotation=30)

    if has_too_many_models:
      plt.xticks([])

    plt.show()

  for metric in metrics:
    plt.figure(figsize=(16, 8))
    data = [getattr(summary[model], metric).values for model in models]
    plt.boxplot(data, tick_labels=models, patch_artist=True, boxprops=dict(facecolor='skyblue'))
    plt.title(f'"{metric}" distribution across different models')
    plt.ylabel(metric)
    plt.xticks(rotation=30)

    if has_too_many_models:
      plt.xticks([])

    plt.show()

  if has_too_many_models: return

  for model in models:
    plt.figure(figsize=(16, 8))
    sns.heatmap(summary[model].confusion, annot=True, cmap='Blues')
    plt.title(f'Confusion matrix: "{model}"')
    plt.xticks(rotation=30)
    plt.show()

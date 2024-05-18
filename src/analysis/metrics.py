from src.experiments import ExperimentsSummary
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_metrics(summary: ExperimentsSummary):
  metrics = ['accuracy', 'recall', 'precision']
  models = list(summary.keys())

  for metric in metrics:
    plt.figure(figsize=(10, 5))
    means = [getattr(summary[model], metric).mean for model in models]
    plt.bar(models, means, color='skyblue')
    plt.title(f'Mean {metric} across different models')
    plt.ylabel(metric)
    plt.show()

  for metric in metrics:
    plt.figure(figsize=(10, 5))
    data = [getattr(summary[model], metric).values for model in models]
    plt.boxplot(data, tick_labels=models, patch_artist=True, boxprops=dict(facecolor='skyblue'))
    plt.title(f'{metric} distribution across different models')
    plt.ylabel(metric)
    plt.show()

  for model in models:
    plt.figure(figsize=(10, 5))
    sns.heatmap(summary[model].confusion, annot=True, cmap='Blues')
    plt.title(f'Confusion matrix: {model}')
    plt.show()

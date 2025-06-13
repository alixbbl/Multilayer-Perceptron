import matplotlib.pyplot as plt
from mlp.config import VISU_OUTPUT

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def plot_metric(metric_name: str, metric_history: list, filename: str, legend: str):

    plt.plot(metric_history, label=f"{metric_name} evolution")
    plt.xlabel('Epochs')
    plt.ylabel(f"{metric_name} evolution")
    plt.legend(legend)
    plt.grid(True)
    plt.savefig(VISU_OUTPUT / filename)
    plt.close()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def plot_metric_compare(metric_name: str, data_trainings: dict, filename, legend: str):
    
    plt.figure(figsize=(10, 6))
    
    for model_name, model_data in data_trainings.items():
        loss_data = model_data.get(f"{metric_name}_history", [])
        plt.plot(loss_data, label=f"{model_name} - {metric_name}")
    
    plt.xlabel('Epochs')
    plt.ylabel(f'{metric_name}')
    plt.legend(legend)
    plt.grid(True)
    plt.title(f'{metric_name} Comparison')
    
    filepath = VISU_OUTPUT / filename
    plt.savefig(filepath)
    plt.close()

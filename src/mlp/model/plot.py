import matplotlib.pyplot as plt
from mlp.config import VISU_OUTPUT

def plot_loss(loss_history: list):

    plt.plot(loss_history, label="Loss evolution")
    plt.xlabel('Epochs')
    plt.ylabel('Cost (Loss)')
    plt.legend()
    plt.grid(True)
    filepath = VISU_OUTPUT / f'Model_loss_history.png'
    plt.savefig(filepath)
    # plt.show()

def plot_accuracy(accuracy_history: list):
    
    plt.plot(accuracy_history, label="accuracy evolution")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    filepath = VISU_OUTPUT / f'Model_training_accuracy_history.png'
    plt.savefig(filepath)
    # plt.show()
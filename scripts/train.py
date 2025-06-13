import argparse

from mlp.model.utils import upload_csv
from mlp.model.plot import plot_metric
from mlp.model.mlp import MLP
import mlp.config as config

def main(parsed_args):

    X_train = upload_csv(filepath=config.DATA_PROCESSING_OUTPUT / config.OUTPUT_FILENAMES[0])
    y_train = X_train['Diagnosis']
    X_train.drop(columns='Diagnosis', inplace=True)

    n_inputs = X_train.shape[1]
    hidden_layers = parsed_args.layers
    n_output = 2 if parsed_args.loss == "categoricalCrossentropy" else 1

    mlp = MLP(n_inputs, 
                hidden_layers, 
                n_output=n_output, 
                loss=parsed_args.loss, 
                learning_rate=parsed_args.learning_rate, 
                save_name = parsed_args.save_training
    )
    loss_history, accuracy_history, precision_history, recall_history = mlp.train(
                X_train, 
                y_train,
                parsed_args.epochs,
                parsed_args.batch_size,
    )
    plot_metric("loss", loss_history, 'Model_loss_history.png', legend="Loss history")
    plot_metric("accuracy", accuracy_history, "Model_accuracy_history.png", legend="Accuracy history")
    plot_metric("recall", recall_history, 'Model_recall_history.png', legend="Recall history")
    plot_metric("precision", precision_history, "Model_precision_history.png", legend="Precision history")
    mlp.save_weights(config.MODEL_PARAMETERS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", 
                        type=int, 
                        nargs="+",
                        required=True,
                        help="Number of neurons in each hidden layer. Ex: --hidden_layers 24 24 24."
                        )
    parser.add_argument("--epochs",
                        type=int,
                        default=50,
                        help="Please enter a valid epoch number - integer. Default is 50."
                        )
    parser.add_argument("--loss",
                        type=str,
                        required=True,
                        choices=("binaryCrossentropy",
                                 "categoricalCrossentropy"),
                        help="Please enter a valid loss method : binaryCrossentropy or categoricalCrossentropy"
                        )
    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
                        help="Please enter a valid batch size. Default = 8."
                        )
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.0314,
                        help="Please enter a valid larning rate. Default = 0.0314."
                        )
    parser.add_argument("--save_training",
                        type=str,
                        default=None,
                        help="""
                                Save training results with given model name.
                                Ex: --save_training SGD_lr01
                            """
                        )
    parsed_args = parser.parse_args()
    main(parsed_args)
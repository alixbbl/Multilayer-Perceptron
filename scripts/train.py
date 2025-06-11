import argparse

from mlp.model.utils import upload_csv
from mlp.model.plot import plot_loss, plot_accuracy
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
                learning_rate=parsed_args.learning_rate
    )
    loss_history, accuracy_history = mlp.train(
                X_train, 
                y_train,
                parsed_args.epochs,
                parsed_args.batch_size,
    )
    plot_loss(loss_history)
    plot_accuracy(accuracy_history)
    mlp.save_parameters(config.MODEL_PARAMETERS)


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
    parsed_args = parser.parse_args()
    main(parsed_args)
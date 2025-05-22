import pandas as pd
import numpy as np
from model.utils import upload_csv
import argparse
from model.mlp import MLP
import constants

DATA_DIR = constants.DATA_DIR

# **************************************************** MAIN *************************************************

def main(parsed_args):

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_train = upload_csv(constants.XY_TRAIN_PATH)
    y_train = X_train['Diagnosis']
    X_train.drop(columns='Diagnosis', inplace=True)
    n_inputs = X_train.shape[1]
    hidden_layers = parsed_args.hidden_layers
    mlp = MLP(n_inputs, hidden_layers, n_output=2) # deux neurones en sortie car on utilisera SoftMax pour obtenir des % et pas juste 0 ou 1

    mlp.train(
                X_train, 
                y_train,
                parsed_args.epochs,
                parsed_args.loss,
                parsed_args.batch_size,
                parsed_args.learning_rate
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_layers", 
                        type=int, 
                        nargs="2+",  # accepte un ou plusieurs entiers
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
                        choices=("binaryCrossEntropy",
                                 "categoricalCrossentropy"),
                        help="Please enter a valid loss method : binaryCrossEntropy or categoricalCrossentropy"
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
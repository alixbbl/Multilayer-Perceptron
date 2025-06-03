import pandas as pd
import numpy as np
from model.utils import upload_csv, save_model_weights
from visualization.plot import plot_loss
import argparse
from model.mlp import MLP
import constants
DATA_DIR = constants.DATA_DIR

# **************************************************** MAIN *************************************************

def main():

    Xy_validation = upload_csv(filepath=constants.XY_VALIDATION_PATH)
    y_val = Xy_validation["Diagnosis"]
    X_val = Xy_validation.drop(columns="Diagnosis")
    print(X_val.head)
    print(y_val.head)

    loss_history, accuracy_history = mlp.train(
                X_train, 
                y_train,
                parsed_args.epochs,
                parsed_args.batch_size
    )
    plot_loss(loss_history)
    plot_accuracy(accuracy_history)
    # save_model_weights(weights, X_train.columns)
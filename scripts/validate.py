import pandas as pd
import numpy as np
import ast

import mlp.config as config
from mlp.model.utils import upload_csv
from mlp.model.plot import plot_loss
from mlp.model.mlp import MLP
from mlp.model.metrics import Metrics


def ft_standardize_data_2(X: pd.DataFrame, constants: pd.DataFrame) -> pd.DataFrame:
    """
        Standardizes the dataframe and returns:
        - standardized dataframe
        - means (as a Series)
        - stds (as a Series)
    """
    mean_vector = X.mean()
    std_vector = X.std()
    df_standardized = (X - mean_vector) / std_vector
    return df_standardized

def load_weights_into_MLP(mlp, filepath):
    """
        Adjust the MLP with saved weights and biases.
    """
    with np.load(filepath) as data:
        for i, layer in enumerate(mlp.layers):

            layer.weights = data[f'layer_{i}_weights']
            layer.biases = data[f'layer_{i}_biases']

# **************************************************** MAIN *************************************************

def main():

    Xy_validation = upload_csv(filepath=config.DATA_PROCESSING_OUTPUT / config.OUTPUT_FILENAMES[1])
    y_val = Xy_validation["Diagnosis"]
    X_val = Xy_validation.drop(columns="Diagnosis")
    
    mapping = {'B': 0, 'M': 1}
    y_true = pd.Series(y_val).map(mapping).values
    
    constants_stand = upload_csv(filepath=config.DATA_PROCESSING_OUTPUT / config.STAND_CONSTANTS)
    X_val_stand = ft_standardize_data_2(X_val, constants_stand)

    network_infra = upload_csv(filepath=config.MODEL_OUTPUT / config.NETWORK_MLP).iloc[0]
    mlp_network = ast.literal_eval(network_infra["mlp_network"])
    loss = network_infra['loss']
    learning_rate = network_infra['learning_rate']
    mlp = MLP(mlp_network[0], mlp_network[1:-1], mlp_network[-1], loss, learning_rate)

    load_weights_into_MLP(mlp, filepath=config.MODEL_PARAMETERS)
    
    y_pred = mlp.predict(X_val_stand)
    metrics = Metrics(y_true, y_pred)
    metrics.print_all_metrics()


if __name__ == "__main__":
    main()
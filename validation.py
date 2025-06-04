import pandas as pd
import numpy as np
import ast
from model.utils import upload_csv
from data_processing.utils import write_output_dataset
from visualization.plot import plot_loss
from model.mlp import MLP
import constants

DATA_DIR = constants.DATA_DIR
MODEL_DIR = constants.MODEL_DIR

def calculate_accuracy(y_pred, y_true) -> float:
    """
        Calculate the accuracy of the model on test data

        :param y_pred: Calculated predictions
        :param y_true: True labels 
        :return: accuracy score (0-1)
    """
    correct = sum(1 for pred, true in zip(y_pred, y_true) if pred == true)
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0
    return accuracy

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
            print(f'For layer {i} : weights are {data[f'layer_{i}_weights']}')
            print(f'For layer {i} : biases are {data[f'layer_{i}_biases']}')
            layer.weights = data[f'layer_{i}_weights']
            layer.biases = data[f'layer_{i}_biases']

# **************************************************** MAIN *************************************************

def main():

    Xy_validation = upload_csv(filepath=constants.XY_VALIDATION_PATH)
    y_val = Xy_validation["Diagnosis"]
    X_val = Xy_validation.drop(columns="Diagnosis")
    
    # Preparer la target y pour validation # en binary ici, a completer si one-hot en categorical
    mapping = {'B': 0, 'M': 1}
    y_true = pd.Series(y_val).map(mapping).values
    
    # Standardizer les features X_val
    constants_stand = upload_csv(filepath=DATA_DIR / constants.STAND_CONSTANTS)
    X_val_stand = ft_standardize_data_2(X_val, constants_stand)
    # write_output_dataset(X_val_stand, "X_val_std.csv", DATA_DIR) # verif de la std

    # Recreer un MLP vierge et y mettre les configs du MLP de train
    network_infra = upload_csv(filepath=MODEL_DIR / constants.NETWORK_MLP).iloc[0]
    mlp_network = ast.literal_eval(network_infra["mlp_network"])
    loss = network_infra['loss']
    learning_rate = network_infra['learning_rate']
    mlp = MLP(mlp_network[0], mlp_network[1:-1], mlp_network[-1], loss, learning_rate)

    load_weights_into_MLP(mlp, filepath=constants.MODEL_PARAMETERS)
    
    # Lancement des predictions et test accuracy
    y_pred = mlp.predict(X_val_stand)
    accuracy = calculate_accuracy(y_pred, y_true)
    print(f'Accuracy : {accuracy}')


if __name__ == "__main__":
    main()
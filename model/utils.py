import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def print_network_structure(mlp_network, width=120):
    border = "*" * width
    empty_line = "*" + " " * (width - 2) + "*"
    message = f"You are building a network with these layers : {mlp_network}"
    centered_message = "* " + message.center(width - 4) + " *"
    print(border)
    print(empty_line)
    print(centered_message)
    print(empty_line)
    print(border)

def upload_csv(filepath: str) -> pd.DataFrame:
    """
    This function loads a CSV file from the given path and returns a DataFrame.

    :param filepath: str - The path to the CSV file.
    :return: pd.DataFrame - The loaded DataFrame.
    """
    try:
        data = pd.read_csv(filepath, encoding='utf-8')
        if data.empty:
            raise ValueError("The CSV file is empty.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except pd.errors.ParserError:
        raise ValueError("Error parsing the CSV file. Check its format.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")

# MODIFIER LA FONCTION POUR SAUVEGARDER LES WEIGHTS APPRIS
def save_model_weights(weights: list, feature_names: list):
    """
        Save model weights and parameters for the prediction phase.
    """    
    print(feature_names)
    # utiliser np.save() pour print les poids sous forme de tableau 
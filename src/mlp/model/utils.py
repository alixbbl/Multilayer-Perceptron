import pandas as pd
import csv

from typing import List
from mlp.config import MODEL_OUTPUT

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


def save_config(mlp_network: List[int], loss: str, learning_rate: float, path=None) -> None:
    
    path = MODEL_OUTPUT / "network_structure.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["mlp_network", "loss", "learning_rate"])
        writer.writerow([str(mlp_network), loss, learning_rate])


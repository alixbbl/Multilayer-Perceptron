import csv
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt


def upload_csv(filepath: str) -> pd.DataFrame:
    """
    This function loads a CSV file from a given path and returns a DataFrame.
    """
    if not filepath.endswith('.csv'):
        raise ValueError("Invalid file format. Please provide a CSV file.\n")
    try:
        data = pd.read_csv(filepath, encoding='utf-8', header=None)
        if data.empty:
            raise ValueError("The CSV file is empty.")
        print(f"Successfully loaded a CSV file of dimensions: {data.shape}!")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty or unreadable.")
    except pd.errors.ParserError:
        raise ValueError("Error parsing the CSV file. Check its format.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")

def display_correlation_matrix(data: pd.DataFrame)-> None:
    """
        Displays a correlation matrix to visualize correlation between courses.    
    """
    data_num = data.select_dtypes(include=['int', 'float'])
    if 'Index' in data_num.columns:
        data_num.drop('Index', axis=1, inplace=True) 
    corr_matrix = data_num.corr().abs()
    plt.figure(figsize=(24, 20))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

def write_output_dataset(dataset: pd.DataFrame, filename: str, directory_path: str)-> None:
    """
    This function writes a Dataset into a CSV file.
    """
    output_path = directory_path / filename
    with open(output_path, mode='w', newline='') as file:
        dataset.to_csv(output_path, index=False)
    print(f"{filename} is printed!")

import math
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from mlp.config import VISU_OUTPUT

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
    VISU_OUTPUT.mkdir(parents=True, exist_ok=True)
    data_num = data.select_dtypes(include=['int', 'float'])
    if 'Index' in data_num.columns:
        data_num.drop('Index', axis=1, inplace=True) 
    corr_matrix = data_num.corr().abs()
    
    plt.figure(figsize=(24, 20))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig(f"{VISU_OUTPUT}/correlation_matrix.png")
    plt.show()

def write_output_dataset(dataset: pd.DataFrame, filename: str, directory_path: str)-> None:
    """
    This function writes a Dataset into a CSV file.
    """
    output_path = directory_path / filename
    with open(output_path, mode='w', newline='') as file:
        dataset.to_csv(output_path, index=False)
    print(f"{filename} is printed!")


def display_histogram(data: pd.DataFrame, option) -> None:
    
    VISU_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    features = [col for col in data.columns if col != "Diagnosis"]
    total_plots = len(features)
    ncols = 4
    nrows = math.ceil(total_plots / ncols)
    
    figsize = (ncols * 4, nrows * 3)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.tight_layout(pad=3.0)
    
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]
    
    for i, feature in enumerate(features):
        ax = axes[i // ncols][i % ncols]
        for feat in data["Diagnosis"].unique():
            feat_data = data[data["Diagnosis"] == feat]
            sns.histplot(feat_data[feature], bins=20, label=feat, ax=ax)
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel("Frequency", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.legend(title="Malignant vs Benign", fontsize=7, title_fontsize=8)
    
    for j in range(total_plots, nrows * ncols):
        fig.delaxes(axes[j // ncols][j % ncols])
    
    fig.suptitle(f"M/B distribution for {option}.", fontsize=12)
    
    filename = VISU_OUTPUT / f"MB_nucleus_histogram_{option}.png"
    plt.savefig(filename)
    plt.show()


def corr_pair_plot_display(df: pd.DataFrame, directory: str, filename: str):
    
    display_correlation_matrix(df)
    sns.pairplot(df, hue='Diagnosis')
    plt.savefig(f"{directory}/{filename}")


def remove_correlated_features(data):
    """
    Calculate a correlation matrix and suppress the correlated features if above threshold (80%).
    """
    corr_matrix = data.corr().abs()
    features_to_keep = list(data.columns)
    features_to_remove = []
 
    for i, feature1 in enumerate(data.columns):
        for j, feature2 in enumerate(data.columns):
            if i >= j:
                continue
            correlation = corr_matrix.loc[feature1, feature2]
            if correlation > 0.8:
                if feature2 not in features_to_remove:
                    features_to_remove.append(feature2)
    final_features = [f for f in features_to_keep if f not in features_to_remove]
    return final_features

def keep_best_features(data, data_features, TARGET):
    """
    Allow to keep only the best features aka the most different ones, based on the means 
    and std.
    """
    target_values = data[TARGET]  # 0 = Benign, 1 = Malignant
    
    feature_scores = {}
    
    for feature in data_features.columns:
        feature_data = data_features[feature]
        
        mean_benign = feature_data[target_values == 0].mean()
        mean_malignant = feature_data[target_values == 1].mean()
        difference = abs(mean_malignant - mean_benign)
        std_dev = feature_data.std()
        normalized_score = difference / (std_dev + 1e-6)
        feature_scores[feature] = normalized_score
        print(f"    {feature}: B={mean_benign:.2f}, M={mean_malignant:.2f}, Score={normalized_score:.2f}")
    
    sorted_features = sorted(feature_scores.items(), 
                           key=lambda x: x[1], reverse=True)

    n_features_to_keep = min(12, len(data.columns))
    best_features = [feature for feature, score in sorted_features[:n_features_to_keep]]
    print(f"🏆 Top 5 features: {[feat for feat, _ in sorted_features[:5]]}")
    
    return best_features
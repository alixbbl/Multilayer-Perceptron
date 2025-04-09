import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from data_processing.utils import upload_csv, display_correlation_matrix, write_output_dataset
from typing import List, Dict, Tuple
from constants import YES, NO, FULL, RELEVANT, COLUMNS_NAMES, INDEX_NAME
from constants import TEST_TRAIN_SPLIT_PARAMS, TARGET, RELEVANT_FEATURES
from constants import OUTPUT_FILENAMES

class dataProcesser():
    
    def __init__(self, df_original):
        self.dataframe = df_original
        self.clean_df = pd.DataFrame() # df vide
        self.ft_set_index()
        self.ft_imputation()
        self.test_size = TEST_TRAIN_SPLIT_PARAMS['test_size']
        self.random_state = TEST_TRAIN_SPLIT_PARAMS['random_state']
        self.shuffle = TEST_TRAIN_SPLIT_PARAMS['shuffle']
    
    def ft_set_index(self):
        """
            Setting index according to constants file.
        """
        self.clean_df = self.dataframe.copy()
        self.clean_df.columns = COLUMNS_NAMES
        self.clean_df.set_index(INDEX_NAME, inplace=True)

    def ft_imputation(self):
        """
            Imputation by mean if some null entries are found in the dataset.
        """
        total_null = 0
        for column in self.clean_df.columns:
            null_sum = self.clean_df[column].isnull().sum()
            total_null+= null_sum
            if null_sum:
                print(f'Nombre de NULL dans {column} est {null_sum}')
                if self.clean_df[column].dtype in [np.int64, np.float64]:
                    mean_serie_value = self.clean_df[column].mean()
                    self.clean_df[column].fillna(mean_serie_value)
                    print(f'Null entries in this dataset were replaced by \
                          the column mean value {mean_serie_value}.')
        if not total_null:
            print("No missing values found in the dataset.")

    def ft_visualize_data(self)-> None:
        """
            This functions helps visualize and understand the use of data, after printing a 
            correlation matrix.
        """
        while True:
            visu_choice = input("Enter a data visualization mode 'FULL' or 'RELEVANT':\n ")
            if visu_choice == FULL:
                display_correlation_matrix(self.clean_df)
            elif visu_choice == RELEVANT:
                sns.pairplot(self.clean_df[['Worst Area', 'Worst Smoothness', 'Mean Texture', 'Diagnosis']], hue='Diagnosis')
                plt.show()
                break
            else:
                print("Not a reasonable choice bro !")

    def ft_train_test_split(self, X_feat: List[str], y_feat: str)-> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
            This function splits the relevant dataset in two unequale parts used 
            for training and testing the model.
            Works randomly. Returns X_train, X_test, y_train, y_test dataframes.
        """
        X = self.clean_df[X_feat]
        y = self.clean_df[y_feat]

        np.random.seed(self.random_state) # on fixe la graine aleatoire - meme graine == meme reproductibilite
        data_size = len(X)
        test_size = int(data_size * self.test_size) # on a fixe la taille du dataset de test a 25% du dataset 
        
        if self.shuffle:
            indices = np.random.permutation(data_size) # va creer un tableau des donnees avec des indices aleatoires
        else:
            indices = np.arange(data_size) # pareil sans le shuffle
        test_indices = indices[:test_size] # on prend les indices de 0 a taille du test
        train_indices = indices[test_size:] # on prend les indices restants
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices] # creation des datasets
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        
        return X_train, X_test, y_train, y_test

# **************************** MAIN *******************************

def main(parsed_args):
    try:
        data=upload_csv(parsed_args.path_csv_to_read)        
        data_processer = dataProcesser(data)
        # data_processer.ft_visualize_data()
        train_test_tuple = data_processer.ft_train_test_split(RELEVANT_FEATURES, TARGET)
        
        for dataframe, filename in zip(train_test_tuple, OUTPUT_FILENAMES):
            write_output_dataset(dataframe, filename)

    except KeyboardInterrupt:
        print("\nOh, you just press CTRL+C... Ciao !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_csv_to_read',
                        type=str,
                        default='./data/data.csv',
                        help="""CSV file to read""")
    parsed_args=parser.parse_args()
    main(parsed_args)
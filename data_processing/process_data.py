import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from data_processing.utils import upload_csv, display_correlation_matrix, write_output_dataset
from typing import List, Dict, Tuple
from constants import FULL, RELEVANT, COLUMNS_NAMES, INDEX_NAME
from constants import TEST_TRAIN_SPLIT_PARAMS, TARGET
from constants import OUTPUT_FILENAMES

class dataProcesser():
    
    def __init__(self, df_original):
        self.dataframe = df_original
        self.clean_df = pd.DataFrame() # df vide
        self.ft_set_index()
        self.ft_imputation()
        self.relevant_features = []
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
                    self.clean_df[column] = self.clean_df[column].fillna(mean_serie_value)
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

    def ft_select_relevant_features(self):
        """
        This function uses the correlation matrix to select the non-correlated then relevant features of the dataset.
        """
        data_num = self.clean_df.select_dtypes(include=['int', 'float'])
        if 'Index' in data_num.columns:
            data_num.drop('Index', axis=1, inplace=True) 
        corr_matrix = data_num.corr().abs()
        threshold = 0.8 # on fixe un seuil pour la correlation excessive
        corr_pairs = corr_matrix.unstack() # on deplie la matrice de correlation en series de paires a deux index (comme tableau excel)
        strong_corrs = corr_pairs[(abs(corr_pairs) > threshold) & (corr_pairs != 1)] # on identifie toutes les paires de fortes correlations

        to_keep = set(COLUMNS_NAMES) - {INDEX_NAME}
        to_drop = set() 
        for (feature1, feature2), value in strong_corrs.items():
            if feature1 not in to_drop and feature2 not in to_drop:
                to_drop.add(feature2)
        to_keep -= to_drop
        self.relevant_features = [ele for ele in to_keep]

        print(f"These are the features relevant for training phase :\n {self.relevant_features}")

    # Utilisation de apply() qui permet de calculer colonne a colonne sans boucle for
    def ft_standardize_data(self, X_train: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """"
            This functions takes a matrix and returns a tuple of calculated elements : 
            the matrix std and mean, and the standardized matrix to operate. 
        """
        X_train_std = X_train.std() # on applique la fonction a chaque colonne et retourne une Serie des std de chaque colonne
        X_train_mean = X_train.mean() # idem avec les moyennes de chaque col. apply() agir comme une boucle for
        training_data_standardized = (X_train - X_train_mean) / X_train_std
        return (X_train_std, X_train_mean, training_data_standardized)
    
    def ft_train_test_split(self, y_feat: str) -> None:
        """
        This function splits the relevant dataset into two unequal parts used 
        for training and testing the model.
        Works randomly. Saves X_train, X_test, y_train, y_test dataframes as CSV files.
        """
        dataset = self.clean_df[self.relevant_features]

        np.random.seed(self.random_state)
        data_size = len(dataset)
        test_size = int(data_size * self.test_size)

        if self.shuffle:
            indices = np.random.permutation(data_size)
        else:
            indices = np.arange(data_size)

        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        X_train = dataset.iloc[train_indices, :-1]
        y_train = dataset.iloc[train_indices, -1]
        X_test = dataset.iloc[test_indices, :-1]
        y_test = dataset.iloc[test_indices, -1]

        Xy_train = pd.concat([X_train, y_train], axis=1)
        X_train_std, X_train_mean, Xy_train_stand = self.ft_standardize_data(Xy_train)

        Xy_test = pd.concat([X_test, y_test], axis=1)
        Xy_test = Xy_test.drop(columns=TARGET)
        write_output_dataset(Xy_train_stand, OUTPUT_FILENAMES[0])
        write_output_dataset(Xy_test, OUTPUT_FILENAMES[1])
        
        standard_const_df = pd.DataFrame({
            "Mean": X_train_mean,
            "Std": X_train_std
        })
        write_output_dataset(standard_const_df, "constants_stand.csv")

        print("Data has been split and saved as Xy_train.csv and Xy_test.csv.")


# **************************** MAIN *******************************

def main(parsed_args):
    try:
        data=upload_csv(parsed_args.path_csv_to_read)        
        data_processer = dataProcesser(data)
        # data_processer.ft_visualize_data() # a remettre pour le projet push
        data_processer.ft_select_relevant_features()
        data_processer.ft_train_test_split(TARGET)

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

# TO DO 
# Erreur due a la standardization de XTrain mais qui contient encore B ou M => a changer
# faire les tests pour voir si les deux datasets crrees et les features sont OK 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from typing import Tuple
from mlp.config import VISU_OUTPUT, DATA_PROCESSING_OUTPUT

from mlp.data_processing.utils import (
    upload_csv,
    write_output_dataset,
    display_histogram,
    corr_pair_plot_display,
    remove_correlated_features,
    keep_best_features,
)

from mlp.config import (
    COLUMNS_NAMES, INDEX_NAME,
    TEST_TRAIN_SPLIT_PARAMS, TARGET,
    OUTPUT_FILENAMES, DATA_DIR,
    LIST_RELEVANT_HISTO, LIST_THREE_FEAT
)

class dataProcesser():

    def __init__(self, df_original, pipeline):
        self.dataframe = df_original
        self.pipeline = pipeline
        self.clean_df = pd.DataFrame()
        self.relevant_features = []

        self.test_size = TEST_TRAIN_SPLIT_PARAMS['test_size']
        self.random_state = TEST_TRAIN_SPLIT_PARAMS['random_state']
        self.shuffle = TEST_TRAIN_SPLIT_PARAMS['shuffle']

        self.ft_set_index()
        self.ft_imputation()


    def ft_set_index(self):
        """
        Set index and rename columns using constants.
        """
        self.clean_df = self.dataframe.copy()
        self.clean_df.columns = COLUMNS_NAMES
        self.clean_df.set_index(INDEX_NAME, inplace=True)


    def ft_imputation(self):
        """
        Impute missing values using mean for numeric columns.
        """
        total_null = 0
        for column in self.clean_df.columns:
            null_sum = self.clean_df[column].isnull().sum()
            total_null += null_sum
            if null_sum and self.clean_df[column].dtype in [np.int64, np.float64]:
                mean_val = self.clean_df[column].mean()
                self.clean_df[column].fillna(mean_val, inplace=True)
                print(f"Missing values in {column} replaced by mean: {mean_val:.3f}")
        if total_null == 0:
            print("No missing values found in the dataset.")


    def ft_standardize_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Standardize the dataframe.
        """
        mean_vec = df.mean()
        std_vec = df.std()
        df_std = (df - mean_vec) / std_vec
        return df_std, mean_vec, std_vec

    def ft_visualize_data(self) -> None:
        """
        Automatically visualize depending on method.
        """
        if self.pipeline == "three":
            df_subset = self.clean_df[self.relevant_features + ['Diagnosis']]
            display_histogram(df_subset, option='Study_features')

        elif self.pipeline == "all":
            df_subset = self.clean_df[self.relevant_features + ['Diagnosis']]
            corr_pair_plot_display(df_subset, VISU_OUTPUT, filename="all_features_pairplot.png")

        else:
            df_subset = self.clean_df[self.relevant_features + ['Diagnosis']]
            display_histogram(df_subset, option='Selected Features')
            corr_pair_plot_display(df_subset, VISU_OUTPUT, filename="my_selected_features_pairplot.png")


    def ft_select_relevant_features(self):
        """
        Select relevant features based on correlation or predefined histogram list.
        """
        data_num = self.clean_df.select_dtypes(include=['int', 'float'])
        if 'Index' in data_num.columns:
            data_num.drop('Index', axis=1, inplace=True)

        if self.pipeline == "three":
            self.relevant_features = LIST_THREE_FEAT
            print(self.relevant_features)
        
        elif self.pipeline == "all":
            to_remove = ['ID', TARGET]
            all_features = [feature for feature in COLUMNS_NAMES if feature not in to_remove]
            self.relevant_features = all_features

        else:
            print(f"Step 1: Relevant features according to histogramme : {len(LIST_RELEVANT_HISTO)}")
            candidate_features = LIST_RELEVANT_HISTO.copy()
            print("Step 2: Suppressing correlated features extracted from histo listing ...")
            final_features = remove_correlated_features(data_num[candidate_features])
            
            if len(final_features) > 15:
                print("Step 3: TOO MUCH FEATURES, let's keep only the best ones...")
                final_features = keep_best_features(data_num, data_num[final_features], TARGET)
                print(f"   => {len(final_features)} features kept !")
                
            self.relevant_features = sorted(final_features)
            print(f"🏆 Top selected {len(self.relevant_features)} features for training :") 
            for feature_name in self.relevant_features:
                print(feature_name)


    def ft_train_test_split(self, y_feat: str) -> None:
        """
        Split the dataset into train/test and standardize features.
        """
        self.ft_select_relevant_features()
        self.ft_visualize_data()
        dataset = self.clean_df[self.relevant_features + [y_feat]]
        np.random.seed(self.random_state)
        data_size = len(dataset)
        test_size = int(data_size * self.test_size)

        indices = np.random.permutation(data_size) if self.shuffle else np.arange(data_size)
        test_idx = indices[:test_size]
        train_idx = indices[test_size:]

        Xy_train = dataset.iloc[train_idx]
        Xy_test = dataset.iloc[test_idx]

        X_train = Xy_train.drop(columns=[y_feat])
        y_train = Xy_train[y_feat]
        X_train_std, mean_vec, std_vec = self.ft_standardize_data(X_train)

        Xy_train_std = X_train_std.copy()
        Xy_train_std[y_feat] = y_train.values

        const_df = pd.DataFrame({
            "Mean": mean_vec,
            "Std": std_vec
        })

        write_output_dataset(const_df, "constants_stand.csv", DATA_PROCESSING_OUTPUT)
        write_output_dataset(Xy_train_std, OUTPUT_FILENAMES[0], DATA_PROCESSING_OUTPUT)
        write_output_dataset(Xy_test, OUTPUT_FILENAMES[1], DATA_PROCESSING_OUTPUT)

        print("Train/test datasets and standardization constants saved.")


# **************************** MAIN *******************************

def main(parsed_args):

    try:
        data = upload_csv(parsed_args.path_csv_to_read)
        pipeline = parsed_args.select_features

        processor = dataProcesser(data, pipeline)
        processor.ft_train_test_split(TARGET)

    except KeyboardInterrupt:
        print("\nExecution interrupted. Goodbye!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_csv_to_read',
                        type=str,
                        default='./data/data.csv',
                        help="CSV file to read")
    
    parser.add_argument('--select_features',
                        type=str,
                        choices={'three', 'all', 'select'},
                        required=True,
                        help="""
                                Select a pipeline/feature type : three features like the research paper model, 
                                all 30 features or selected ones by combined approach (histogramme, ...).
                            """)

    args = parser.parse_args()
    main(args)

import os
import zipfile
from MA.technical_indicators import *


def extract_zip(path) -> list:
    """
    Extract a zip file
    :param path: path of the zip file
    :return: list containing extracted filenames
    """
    archive = zipfile.ZipFile(path, 'r')
    archive.extractall('data')
    return archive.namelist()


def load_pklfile(path) -> pd.DataFrame:
    """
    load pickle file from path
    :param path: path of pickle file
    :return: (df) pandas dataframe
    """
    df = pd.read_pickle(path)
    return df


def training_test_split(df) -> tuple:
    """
    split dataset into training and testing set
    :param df: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    end = '2017-01-01'  # TODO: Adjust cut off for training/test set split
    training_set_bool_array = pd.Series(df.index < end, name='bools')
    training_set = df[training_set_bool_array.values]

    test_set_bool_array = pd.Series(df.index >= end, name='bools')
    test_set = df[test_set_bool_array.values]

    return training_set, test_set


def preprocess_data() -> pd.DataFrame:
    """
    processing data
    :return: (df) pandas dataframe
    """
    file_list = extract_zip('data/Intraday_Data.zip')
    globex_code = ['ES', 'ZN']  # Globex code of the used future contracts
    datasets = []
    for ind, file in enumerate(file_list):
        data = load_pklfile('data/'+str(file))

        # run technical indicators on the current data
        data = create_tech_indicators(data)

        # drop unnecessary columns (ticker)
        data = data.drop('Ticker', axis=1)

        # rename columns to distinguish between the asset columns
        data.columns = [f'{globex_code[ind]}:{c.lower()}' for c in data.columns]
        datasets.append(data)

    # merge data sets from datasets list to one big dataset on datetime index
    final_df = pd.concat(datasets, axis=1)

    # TODO: drop nans? some data points are available for one future contract but not the other
    # The nans at the end are missing due to a shorter data set -> berücksichtigen in the code
    """    
    # print(final_df.iloc[0])
    # print(final_df.iloc[-1])
    # final_df = final_df.dropna(axis=0)
    # print('og shape: 263551, 44', final_df.shape, final_df.columns,)
    # print('dropped', 263551 - final_df.shape[0], 'rows')
    # print(final_df.iloc[-1])
    """

    # return training & test data
    return final_df


def run_preprocess(data_path) -> tuple:
    """Run the preprocessing of the data"""

    if os.path.exists(data_path):
        processed_data = load_pklfile(data_path)
    else:
        processed_data = preprocess_data()
        # processed_data.to_pickle(preprocessed_data)  # Uncomment line to create data file
        # processed_data.to_csv("data/processed_data.csv")  # for data inspection

    # training test split
    return training_test_split(processed_data)

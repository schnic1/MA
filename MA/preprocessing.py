import os
import zipfile
import pandas as pd
import math
import numpy as np

from MA.technical_indicators import create_tech_indicators
from MA.config import ZIP_PATH, CUT_OFF_DATE


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


def training_test_split(df, cut_off, date_col='date') -> tuple:
    """
    split dataset into training and testing set
    :param df: (df) pandas dataframe
    :param cut_off:  cut-off date to split into training and test set
    :param date_col: default date columns used for splitting
    :return: (df) pandas dataframe
    """
    training_set = df[(df[date_col]) < cut_off]
    test_set = df[(df[date_col]) >= cut_off]

    training_set = training_set.sort_values([date_col, 'ticker'], ignore_index=True)
    test_set = test_set.sort_values([date_col, 'ticker'], ignore_index=True)

    training_set.index = training_set[date_col].factorize()[0]
    test_set.index = test_set[date_col].factorize()[0]

    return training_set, test_set


def preprocess_data(zip_path) -> pd.DataFrame:
    """
    processing data
    :return: (df) pandas dataframe
    """
    file_list = extract_zip(zip_path)
    globex_code = ['ES', 'ZN']  # Globex code of the used future contracts
    datasets = []
    for ind, file in enumerate(file_list):
        data = load_pklfile('data/'+str(file))

        # add 'date' column to dataframe and reset index
        data.insert(loc=0, column='date', value=data.index)
        data.reset_index(inplace=True)
        data = data.drop('index', axis=1)  # drop index column which at this point is still the datetime index

        # run technical indicators on the current data
        data = create_tech_indicators(data)
        datasets.append(data)

    # extract all the unique dates from both dataframes
    unique_dates = pd.concat([datasets[0], datasets[1]])['date'].unique()
    unique_dates_df = pd.DataFrame({'date': unique_dates})

    # add the missing dates from the other dataframe
    full_dfs = []
    for ind, dataset in enumerate(datasets):
        # extend dataframe by missing dates
        all_dates_dataframe = unique_dates_df.merge(dataset, how='outer')

        # rename columns to all lower letters
        all_dates_dataframe.columns = [f'{c.lower()}' for c in all_dates_dataframe.columns]
        all_dates_dataframe = all_dates_dataframe.assign(ticker=globex_code[ind])
        full_dfs.append(all_dates_dataframe)

    # merge the equally long dataframe
    final_df = pd.merge(full_dfs[0], full_dfs[1], how='outer')
    final_df = final_df.sort_values(['date', 'ticker']).reset_index(drop=True)
    date_tic = final_df[final_df.columns.tolist()[:2]]
    other_cols = final_df[final_df.columns.tolist()[2:]]
    other_cols = other_cols.astype('object').fillna(0).astype('float')
    final_df = date_tic.join(other_cols)

    return final_df


def run_preprocess(data_path) -> tuple:
    """
    Run the preprocessing of the data
    :return: training and testing (df) pandas dataframes
    """

    if os.path.exists(data_path):
        processed_data = load_pklfile(data_path)
        print('loaded')
    else:
        print('starting preprocessing')
        processed_data = preprocess_data(ZIP_PATH)
        # processed_data.to_pickle(data_path)  # Uncomment line to create data file
        # processed_data.to_csv("data/test_file.csv")  # for data inspection

    # training test split
    training, test = training_test_split(processed_data, CUT_OFF_DATE)

    return training, test, processed_data

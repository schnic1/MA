import zipfile
import pandas as pd
from MA.technical_indicators import *
import os


#  and return
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

#def rename_columns(df) -> pd.DataFrame:



def training_test_split(df) -> pd.DataFrame:
    """
    split dataset into training and testing set
    :param df: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    end = '2017-01-01'  #TODO: Adjust cut off for training/test set split
    training_set = df.index < end
    test_set = df.index > end
    return training_set, test_set



def preprocess_data()-> pd.DataFrame:
    """
    processing data
    :return: training and test dataset
    """
    file_list = extract_zip('data/Intraday_Data.zip')
    globex_code = ['ES', 'ZN']  # Globex code of the used future contracts
    datasets = []
    for ind, file in enumerate(file_list):
        data = load_pklfile('data/'+str(file))

        #TODO: drop unnecessary columns like ticker
        data = create_tech_indicators(data)

        # rename columns to distinguish between the assets
        data.columns = [f'{globex_code[ind]}:{c.lower()}' for c in data.columns]
        datasets.append(data)

        # print(data.columns)
        # print((data['ES:bb_bbhi']==1).sum(), (data['ES:bb_bbhi']==0).sum())

    # merge data sets from datasets list to one big dataset on datetime index
    final_df = pd.concat(datasets, axis=1)
    #print(final_df.iloc[-1])
    final_df = final_df.dropna(axis=0)
    # print('og shape: 263551, 44', final_df.shape, final_df.columns,)
    # print('dropped', 263551 - final_df.shape[0], 'rows')
    #print(final_df.iloc[-1])
    # print(final_df.dtypes)

    # TODO: check whether dates are congruent in the both datasets



    # split into training & test data

    # return training & test data








import os
import zipfile
import pandas as pd

from MA.technical_indicators import create_tech_indicators
from MA.config import ZIP_PATH, CUT_OFF_DATE_train, CUT_OFF_DATE_test


def extract_zip(path) -> list:
    """
    Extract a zip file
    :param path: path of the zip file
    :return: list containing extracted filenames
    """
    archive = zipfile.ZipFile(path, 'r')
    archive.extractall('data')
    return archive.namelist()


def load_pkl_file(path) -> pd.DataFrame:
    """
    load pickle file from path
    :param path: path of pickle file
    :return: (df) pandas dataframe
    """
    df = pd.read_pickle(path)
    return df


def standardize_data(training, validation, test, tech_indicator_list) -> tuple:
    """
    standardize technical indicator for the agent
    # TODO: del source again:
    Source: https://towardsdatascience.com/preventing-data-leakage-in-your-machine-learning-model-9ae54b3cd1fb#:~:text=After%20splitting%20the%20data%20into,validation%20and%20test%20set%20normalization.
    :param training: (df) pandas dataframe
    :param validation: (df) pandas dataframe
    :param test: (df) pandas dataframe
    :param tech_indicator_list: list of technical indicators
    :return: tuple with standardized training, validation and test set
    """
    for tech_ind in tech_indicator_list:
        # only standardize for non-binary indicators
        if len(training[tech_ind].unique()) > 2:
            # determine mean and standard deviation for standardization from training set
            mean = training[tech_ind].mean()
            std = training[tech_ind].std()

            # standardize sets with training mean and standard deviation
            training[tech_ind] = (training[tech_ind] - mean)/std
            validation[tech_ind] = (validation[tech_ind] - mean)/std
            test[tech_ind] = (test[tech_ind] - mean)/std

    return training, validation, test


def training_test_split(df, cut_off_training, cut_off_test, date_col='date') -> tuple:
    """
    split dataset into training and testing set
    :param df: (df) pandas dataframe
    :param cut_off_training:  cut-off date to split into training & validation set
    :param cut_off_test:  cut-off dates to split into validation & test set
    :param date_col: default date columns used for splitting
    :return: (df) pandas dataframe
    """
    training_set = df[(df[date_col] < cut_off_training)]
    validation_set = df[(df[date_col] >= cut_off_training) & (df[date_col] < cut_off_test)]
    test_set = df[(df[date_col] >= cut_off_test)]

    training_set = training_set.sort_values([date_col, 'ticker'], ignore_index=True)
    validation_set = validation_set.sort_values([date_col, 'ticker'], ignore_index=True)
    test_set = test_set.sort_values([date_col, 'ticker'], ignore_index=True)

    training_set.index = training_set[date_col].factorize()[0]
    validation_set.index = validation_set[date_col].factorize()[0]
    test_set.index = test_set[date_col].factorize()[0]

    return training_set, validation_set, test_set


def align_and_merge(dataset_list, globex_code_list) -> pd.DataFrame:
    """
    add missing timestamps to the dataframes and combine in the same time interval
    :param dataset_list: list containing the dataset for both tickers
    :param globex_code_list: contains the globex code of the future contracts
    :return: (df) pandas dataframe; merged with entries for both ticker for each timestamp
    """
    # extract all unique timestamps from datasets
    unique_dates = pd.concat([dataset_list[i] for i in range(len(dataset_list))])['date'].unique()
    unique_dates_df = pd.DataFrame({'date': unique_dates})

    all_dates_dfs = []
    for ind, dataset in enumerate(dataset_list):
        # extend dataset by missing timestamps
        all_dates = unique_dates_df.merge(dataset, how='outer')

        # set ticker for the dataframe filling newly added lines
        all_dates = all_dates.assign(ticker=globex_code_list[ind])
        all_dates_dfs.append(all_dates)

    # combine dataframes with the different tickers but same timeinterval
    full_df = pd.merge(all_dates_dfs[0], all_dates_dfs[1], how='outer')
    full_df = full_df.sort_values(['date', 'ticker']).reset_index(drop=True)
    full_df = full_df.fillna(0)

    return full_df


def preprocess_data(zip_path) -> list:
    """
    processing data
    :return: (df) pandas dataframe
    """
    file_list = extract_zip(zip_path)
    globex_code = ['ES', 'ZN']  # Globex code of the used future contracts
    training_sets = []
    validation_sets = []
    test_sets = []
    for ind, file in enumerate(file_list):
        data = load_pkl_file('data/'+str(file))

        # add 'date' column to dataframe and reset index
        data.insert(loc=0, column='date', value=data.index)
        data.reset_index(inplace=True)
        data = data.drop('index', axis=1)  # drop index column which at this point is still the datetime index

        # run technical indicators on the current data
        data, tech_ind_list = create_tech_indicators(data)

        # rename columns to all lower
        data.columns = [f'{c.lower()}' for c in data.columns]
        # data = data.assign(ticker=globex_code[ind])

        # split data into training, evaluation & test set
        training, validation, test = training_test_split(data, CUT_OFF_DATE_train, CUT_OFF_DATE_test)

        # standardize technical indicator data using mean and std from training set for all
        training, validation, test = standardize_data(training, validation, test, tech_ind_list)

        # append the standardized sets to the according lists
        training_sets.append(training)
        validation_sets.append(validation)
        test_sets.append(test)

    # the training, validation and test the sets for both tickers, length of sub-lists = number of tickers
    datasets = [training_sets, validation_sets, test_sets]

    final_sets = []
    for dataset_list in datasets:
        aligned = align_and_merge(dataset_list, globex_code)
        aligned.index = aligned['date'].factorize()[0]
        final_sets.append(aligned)

    return final_sets


def run_preprocess(data_path) -> tuple:
    """
    Run the preprocessing of the data
    :return: training and testing (df) pandas dataframes
    """

    if os.path.exists(data_path):
        processed_data = load_pkl_file(data_path)
        # training, validation, test split of loaded and processed data
        training, validation, test = training_test_split(processed_data, CUT_OFF_DATE_train, CUT_OFF_DATE_test)
        print('data loaded')

    else:
        print('starting preprocessing')
        set_list = preprocess_data(ZIP_PATH)
        processed_data = pd.concat([df for df in set_list], axis=0)
        processed_data.to_pickle(data_path)  # Uncomment line to create data file
        processed_data.to_csv("data/test_file.csv")  # for data inspection

        training, validation, test = set_list[0], set_list[1], set_list[2]

    return training, validation, test, processed_data

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
    start = '2010-01-01'
    cut_off = '2017-01-01'
    end = '2022-05-31'
    training_set = data_split(df, start, cut_off)
    test_set = data_split(df, cut_off, end)

    return training_set, test_set


def preprocess_data_old() -> pd.DataFrame:
    """
    processing data
    :return: (df) pandas dataframe
    """
    file_list = extract_zip('data/Intraday_Data.zip')
    globex_code = ['ES', 'ZN']  # Globex code of the used future contracts
    datasets = []
    for ind, file in enumerate(file_list):
        data = load_pklfile('data/'+str(file))

        # TODO: delete to use whole dataset!
        # data = data.head(500)

        # run technical indicators on the current data
        data = create_tech_indicators(data)

        # drop unnecessary columns (ticker)
        data = data.drop('Ticker', axis=1)

        # rename columns to distinguish between asset columns
        data.columns = [f'{globex_code[ind]}:{c.lower()}' for c in data.columns]

        datasets.append(data)

    final_df = pd.concat(datasets, axis=1)
    # return training & test data
    print(final_df.head(10))
    return final_df


def preprocess_data() -> pd.DataFrame:
    file_list = extract_zip('data/Intraday_Data.zip')
    globex_code = ['ES', 'ZN']  # Globex code of the used future contracts
    datasets = []
    for ind, file in enumerate(file_list):
        data = load_pklfile('data/'+str(file))

        # add 'date' column to dataframe and reset index
        data.insert(loc=0, column='Date', value=data.index)
        data.reset_index(inplace=True)
        data = data.drop('index', axis=1)  # drop index column which at this point is still the datetime index
        data = data.assign(Ticker=globex_code[ind])

        # run technical indicators on the current data
        data = create_tech_indicators(data)

        # rename columns to all lower letters
        data.columns = [f'{c.lower()}' for c in data.columns]
        datasets.append(data)
        print(data.describe())

    final_df = pd.merge(datasets[0], datasets[1], how='outer')
    final_df = final_df.sort_values(['date', 'ticker']).reset_index(drop=True)
    return final_df


def run_preprocess(data_path) -> tuple:
    """Run the preprocessing of the data"""

    if os.path.exists(data_path):
        processed_data = load_pklfile(data_path)
        print('loaded')
    else:
        print('starting preprocessing')
        processed_data = preprocess_data()
        # processed_data.to_pickle(data_path)  # Uncomment line to create data file
        # processed_data.to_csv("data/test_file.csv")  # for data inspection

    # training test split
    # training, test = training_test_split(data)

    return training, test, processed_data

# start/code for input dimension (2, 23); both contracts per date
# not finished!!!
"""
        # add 'date' column to dataframe and reset index
        data.insert(loc=0, column='Date', value=data.index)
        data.reset_index(inplace=True)
        data = data.drop('index', axis=1)  # drop index column which at this point is still the datetime index
        data = data.assign(Ticker=globex_code[ind])

        # rename columns to all lower letters
        data.columns = [f'{c.lower()}' for c in data.columns]
        datasets.append(data)


    print(len(datasets[0]))  # 260756 datapoints
    print(len(datasets[1]))  # 257769 datapoints

    unique_dates = pd.concat([datasets[0], datasets[1]])['date'].unique()
    print('unique dates: ', len(unique_dates))
    print(type(unique_dates))

    unique_dates_df = pd.DataFrame({'date': unique_dates})
    print(len(unique_dates_df))
    print(unique_dates_df.dtypes)
    hgne = unique_dates_df.merge(datasets[0], how='outer')
    print(hgne)


    # merge data sets from datasets list to one big dataset on date column and reset index
    final_df = pd.merge(datasets[0], datasets[1], how='outer')
    final_df = final_df.sort_values(['date', 'ticker']).reset_index(drop=True)
    """

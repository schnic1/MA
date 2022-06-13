# import statements
import zipfile
import pandas as pd
import os


def extract_zip(filename):
    path = os.path.join('data', filename)
    archive = zipfile.ZipFile(path, 'r')
    archive.extractall('data')
    return archive.namelist()


def open_pkl_file(filename):
    # open pickle file and generate pandas dataframe
    df = pd.read_pickle(filename)
    return df


def mod_ticker_column(df):
    # generate year column for unique ticker column: 'mod_ticker'
    df['Year'] = pd.DatetimeIndex(df.index).year.astype(str)

    # Differentiation between ESH0 in 2010 and 2020
    # Also same modified ticker for futures starting in 2019 but ending in 2020

    # generate modified ticker column
    df['mod_Ticker'] = df['Ticker'] + '_' + df['Year']

    # set same modified ticker for data points of the same futures contract
    for tkr in df['mod_Ticker'].unique():
        if tkr[3] != tkr[-1]:
            indexer = df[df['mod_Ticker'] == tkr].index

            # special case with 9, e.g. 19 ESH0 with 2019 belongs to 2020, not 2010
            if tkr[-1] == '9':
                corr_tkr = tkr[:-2] + str(int(tkr[-2:]) + 1)

            # normal case of mod_ticker correction: take integer in ticker
            else:
                corr_tkr = tkr[:-1] + tkr[3]
            df.loc[indexer, 'mod_Ticker'] = corr_tkr

    return df

def run_data_cleaning(filename):
    df = mod_ticker_column(open_pkl_file(filename))

    return df

if

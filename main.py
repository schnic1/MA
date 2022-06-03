# import statements
from MA.data_processing import *
from MA.technical_indicators import *
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

filenames = extract_zip('Intraday_Data.zip')

for file in filenames:
    # generate file path
    file_path = "data/" + str(file)  # file 1: ES1, file 2: ZN1
    data = run_data_cleaning(file_path)
    data = create_tech_indicators(data)
    print(file, " columns in dataset: ", data.columns)

    # save datasets to data folder
    data.to_pickle('data/ind_'+str(file))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

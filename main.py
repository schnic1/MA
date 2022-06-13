# import statements
from MA.preprocessing import *
from MA.technical_indicators import *
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def run_model() -> None:
    """Train the model"""

    # load and preprocess the data
    preprocessed_data = "data/training.pkl" #TODO: change filename

    if os.path.exists(preprocessed_data):
        data = load_pklfile(preprocessed_data)
    else:
        data = preprocess_data()
        data.to_pickle("data/training.pkl") #TODO: change filename to the one above

run_model()
"""

filenames = extract_zip('Intraday_Data.zip')

for file in filenames:
    # generate file path
    file_path = "data/" + str(file)  # file 1: ES1, file 2: ZN1
    data = run_data_cleaning(file_path)
    data = create_tech_indicators(data)
    print(file, " columns in dataset: ", data.columns)
    # TODO: find nan values in the dataframe

    # save datasets to data folder
    # data.to_pickle('data/ind_'+str(file))
"""
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

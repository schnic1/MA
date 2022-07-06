# import statements
from MA.preprocessing import *
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

training, test = run_preprocess("data/processed_data.pkl")

# test if everything works
print(training.head())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

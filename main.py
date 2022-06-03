# import statements
from MA.data_processing import *
from MA.technical_indicators import *

"""extract_zip('Intraday_Data.zip')

data = open_pkl('ES1 Index.pkl')
print(data.columns)"""

filename = 'data/ES1 Index.pkl'

data = run_data_cleaning(filename)

data = create_tech_indicators(data)

print(len(data.columns))



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

# data file paths for loading and saving
ZIP_PATH = "data/Intraday_Data.zip"
DATA_PATH = "data/processed_data.pkl"

NORM_ROLLING_WINDOW = 100

# environment kwargs
env_kwargs = {"initial_amount": 100000,
              "contract_size": [50, 1000],  # [ES, ZN]
              "margins": [18000, 5000],  # [ES, ZN]
              "bid_ask": [0.5, 0.015],  # [ES, ZN], according to Bloomberg Aug. 2022
              "commission": 2,  # normal market rate
              "validation": False
              }

# data split specifications -> training, validation, test set
CUT_OFF_DATE_train = "2015-01-01"
CUT_OFF_DATE_test = "2020-01-01"

# model parameters to start with
AGENT_PARAM_DICT = {'A2C_PARAMS': {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007},
                    'PPO_PARAMS':  {"n_steps": 256, "ent_coef": 0.01, "learning_rate": 0.0007, "batch_size": 32}
                    }

# training specification
# available agents for method: ["a2c", "ppo"]
method = 'ppo'
run_training = True
# if run_training = False, define model to be loaded from the 'models' folder
trained_model = 'A2C_model_0.zip'

# saving & loading models path
SAVE_MODEL_PATH = "models/"

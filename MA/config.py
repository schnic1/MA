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
CUT_OFF_DATE_train = "2016-01-01"
CUT_OFF_DATE_test = "2020-01-01"

# model parameters to start with
AGENT_PARAM_DICT = {'A2C_PARAMS': {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007},
                    'PPO_PARAMS':  {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 64}
                    }

# training specification
# available agents for method: ["a2c", "ppo"]
method = 'ppo'
run_training = True


# saving & loading models path
SAVE_MODEL_PATH = "models/"
trained_model = 'PPO_val_18_09 00:21.zip'  # give name of model to load
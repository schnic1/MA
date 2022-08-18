# data file paths for loading and saving
ZIP_PATH = "data/Intraday_Data.zip"
DATA_PATH = "data/processed_data.pkl"

# environment kwargs
env_kwargs = {"max_contracts": 10,
              "initial_amount": 100000,
              "buying_fee": 0.001,  # ES: , ZN:
              "selling_fee": 0.001}  # ES: , ZN:

new_env_kwargs = {"initial_amount": 100000,
                  "contract_size": [50, 1000],  # [ES, ZN]
                  "margins": [18000, 5000],  # [ES, ZN]
                  "bid_ask": [0.5, 0.5],  # [ES, ZN], according to Bloomberg Aug. 2022
                  "commission": 2,  # normal market rate
                  "margin_calls": [9000, 2500]  # [ES, ZN]
                  }

# data split specifications -> training, validation, test set
CUT_OFF_DATE_train = "2019-01-01"
CUT_OFF_DATE_test = "2020-01-01"

# model parameters to start with
AGENT_PARAM_DICT = {'A2C_PARAMS': {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007},
                    'PPO_PARAMS':  {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025,
                                    "batch_size": 64},
                    'DDPG_PARAMS': {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001},
                    'TD3_PARAMS': {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001},
                    'SAC_PARAMS': {"batch_size": 64, "buffer_size": 100000, "learning_rate": 0.0001,
                                   "learning_starts": 100, "ent_coef": "auto_0.1"}
                    }

# training specification
# available agents for method: ["a2c", "ddpg", "td3", "sac", "ppo"]
method = 'a2c'

# saving & loading models path
SAVE_MODEL_PATH = "models/"
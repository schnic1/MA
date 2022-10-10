### training specification ###

# available agents for method: ["a2c", "ppo"]
method = 'ppo'
run_training = True
evaluation = False
# if run_training = False, define model to be loaded from the 'models' folder
trained_model = 'PPO_model_60.zip'

# if evaluation True, define period to be evaluated and loaded from test_pred folder
evaluation_period = 'episode_12_steps2140_03_10.csv'
bm_pos = [2, 7]

TOTAL_TIME_STEPS = 1000

# model parameters to start with
AGENT_PARAM_DICT = {'A2C_PARAMS': {"n_steps": 512,
                                   "ent_coef": 0.01,
                                   "learning_rate": 0.0004,
                                   "gamma": 0.9},

                    'PPO_PARAMS': {"n_steps": 2048,
                                   "ent_coef": 0.025,
                                   "learning_rate": 0.0002,
                                   "clip_range": 0.2,
                                   "gamma": 0.85}
                    }

# environment kwargs
env_kwargs = {"initial_amount": 100000,
              "contract_size": [50, 1000],  # [ES, ZN]
              "margins": [18000, 5000],  # [ES, ZN]
              "bid_ask": [0.5, 0.015],  # [ES, ZN], according to Bloomberg Aug. 2022
              "commission": 2,  # normal market rate
              "validation": False,
              'reward_arg': 'return'  # define reward function ['return', 'sharpe']
              }

### program specifications ###
# saving & loading models path
SAVE_MODEL_PATH = "model_data/models/"

# data file paths for loading and saving
ZIP_PATH = "data/Intraday_Data.zip"
DATA_PATH = "data/processed_data.pkl"

# evaluation path
EVAL_PATH = "model_data/episode_data/test_pred/"
PLOT_PATH = "model_data/evaluation/"

# data split specifications -> training, validation, test set
CUT_OFF_DATE_train = "2015-01-01"
CUT_OFF_DATE_test = "2020-01-01"

# look back for rolling window normalization
NORM_ROLLING_WINDOW = 100


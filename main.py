# import statement
import time
import warnings

from datetime import datetime
from dateutil import relativedelta

import pandas as pd

import MA.config
from MA.preprocessing import run_preprocess
from MA.environment_futures import build_env, show_env
from MA.agent import build_agent, train_model, save_model, make_prediction, load_model, policy_evaluation
from MA.config import DATA_PATH, env_kwargs, CUT_OFF_DATE_test, CUT_OFF_DATE_train
from MA.config import method, run_training

# time computation time
start = time.time()
warnings.filterwarnings("ignore", category=RuntimeWarning)

# fetching data
training_set, validation_set, test_set, processed_data = run_preprocess(DATA_PATH)
print(training_set.shape)  # shape = (239670, 23)
print(validation_set.shape)  # shape = (236444, 23)
print(test_set.shape)  # shape = (50988, 23)


# define agent specifications
method = method

if run_training:
    # building training environment
    env, _ = build_env(training_set, env_kwargs)
    # show_env(10, env)  # see how the environment works with actions, step, etc.

    # build agent
    model = build_agent(env, method)

    # define total time steps for training on training set
    total_timesteps = 12000000

    # train & save model
    print('### started training on training set ###')
    trained_model = train_model(model, total_timesteps=total_timesteps)
    model_name = save_model(trained_model, method, printing=True)

    after_training_time = time.time()
    print(f'running time after training:{round((after_training_time - start) / 60, 0)} minutes')

    print('### started on validation set ###')
    val_periods_start = datetime.strptime(CUT_OFF_DATE_train, '%Y-%m-%d')
    val_periods_end = datetime.strptime(validation_set.date[-1:].unique()[0], '%Y-%m-%d %H:%M:%S')
    val_periods = relativedelta.relativedelta(val_periods_end, val_periods_start)
    months = val_periods.years * 12 + val_periods.months + 1 # the +1 is for the started month, which is not included otherwise

    env_kwargs['validation'] = True
    months_in = 0
    initial_validation_date = datetime.strptime(validation_set.date[0].unique()[0], '%Y-%m-%d %H:%M:%S')
    total_timesteps_val = total_timesteps // months

    for period in range(4, months + 1):
        train_start = initial_validation_date + pd.offsets.DateOffset(months=months_in)
        train_end = train_start + pd.offsets.DateOffset(months=3)
        test_end = train_end + pd.offsets.DateOffset(months=1)

        mask_train = (validation_set['date'] >= str(train_start)) & (validation_set['date'] <= str(train_end))
        mask_test = (validation_set['date'] > str(train_end)) & (validation_set['date'] <= str(test_end))
        val_training_set = validation_set.loc[mask_train]
        val_training_set.index = val_training_set['date'].factorize()[0]
        val_prediction_set = validation_set.loc[mask_test]
        val_prediction_set.index = val_prediction_set['date'].factorize()[0]

        # training model
        env, _ = build_env(val_training_set, env_kwargs)
        env.saving_folder = 2
        trained_model = load_model(method, model_name, env)
        trained_model = train_model(trained_model, total_timesteps=total_timesteps_val)
        model_name = save_model(trained_model, method, period=period)

        val_pred_env, _ = build_env(val_prediction_set, env_kwargs)
        val_pred_env.saving_folder = 1
        trained_model = load_model(method, model_name, val_pred_env)
        make_prediction(trained_model, val_pred_env)

        months_in += 1

    model_name_val = save_model(trained_model, method, validation=True)

    after_validation_time = time.time()
    print(f'running time after training:{round((after_validation_time - after_training_time) / 60, 0)} minutes')

else:
    env_kwargs['validation'] = True
    model_name_val = MA.config.trained_model

# test environment
test_env, _ = build_env(test_set, env_kwargs)
test_env.saving_folder = 3
print('predicting with test set')
val_loaded_model = load_model(method, model_name_val, test_env)

make_prediction(val_loaded_model, test_env)

# policy_evaluation(val_loaded_model, test_env)

end = time.time()
print(f'running time:{round((end - start)/60, 0)} minutes')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

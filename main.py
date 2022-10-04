# import statement
import os
import time
import warnings
import os

from datetime import datetime
from dateutil import relativedelta

import pandas as pd
import numpy as np

import MA.config
from MA.preprocessing import run_preprocess
from MA.environment_futures import build_env, show_env
from MA.agent import build_agent, train_model, save_model, make_prediction, load_model, policy_evaluation
from MA.config import DATA_PATH, CUT_OFF_DATE_train
from MA.config import method, run_training, env_kwargs, AGENT_PARAM_DICT, TOTAL_TIME_STEPS, evaluation

from MA.evaluation import run_eval
from MA.config import evaluation

# time computation time
start = time.time()
warnings.filterwarnings("ignore", category=RuntimeWarning)

parent_dirs =['episode_data', 'logs', 'models', 'evaluation']

for dir in parent_dirs:
    os.makedirs(f'model_data/{dir}', exist_ok=True)

child_dirs = ['training', 'test_pred', 'val_pred', 'val_train', 'policy_eval']

for dir in child_dirs:
    os.makedirs(f'model_data/episode_data/{dir}', exist_ok=True)

# fetching data
training_set, validation_set, test_set, processed_data = run_preprocess(DATA_PATH)
print(training_set.shape)  # shape = (239670, 23)
print(validation_set.shape)  # shape = (236444, 23)
print(test_set.shape)  # shape = (50988, 23)

# generate file to write down model specifications
f = open(f'model_data/model_specs.txt', 'w')

# define agent specifications
method = method
f.write(f'method: {method},\n')

# define if run_training in config.py file
# if run_training = False, model zip needs to be given as parameter
if run_training:
    '''
    """
    in the first section, the agent is trained on the training set with random initialisation of point in time and 
    contract position in order to get it to learn different situations in general and thus keeping it flexible.
    """
    # building training environment
    env, _ = build_env(training_set, env_kwargs)
    # show_env(10, env)  # see how the environment works with actions, step, etc.
    f.write(f'reward function: {env.reward_arg} \n')

    # build agent
    agent = build_agent(env, method)

    f.write(f'agent specs: \n')
    for key, value in AGENT_PARAM_DICT[f"{method.upper()}_PARAMS"].items():
        f.write(f'{key}: {value}\n')

    # define total time steps for training on training set
    total_timesteps = TOTAL_TIME_STEPS
    f.write(f'training timesteps: {total_timesteps}\n')

    print('### started training on training set ###')
    trained_agent = train_model(agent, total_timesteps=total_timesteps,)

    model_name = save_model(trained_agent, method, printing=True)

    after_training_time = time.time()
    print(f'running time for training:{round((after_training_time - start) / 60, 0)} minutes')

    """
    After the initial training, the agent now is being retrained for a period of three month before it predicts its 
    trading decisions for the next month, updating the existing model with each new retraining. The training as well as
    the prediction records for each period are saved to the val_train resp. val_pred folder.
    
    The model names for each period are stored in a dataframe to later being evaluated against the other period's
    retrained agents to eventually select the best performing model.
    """
    print('### started on validation set ###')

    # to define how many periods the agent is retrained for, dependent on the validation set time horizon
    val_periods_start = datetime.strptime(CUT_OFF_DATE_train, '%Y-%m-%d')
    val_periods_end = datetime.strptime(validation_set.date[-1:].unique()[0], '%Y-%m-%d %H:%M:%S')
    val_periods = relativedelta.relativedelta(val_periods_end, val_periods_start)

    # the +1 is for the started month, which is not included otherwise
    months = val_periods.years * 12 + val_periods.months + 1

    env_kwargs['validation'] = True  # adjust env_kwargs dict to validation mode

    months_in = 0
    initial_validation_date = datetime.strptime(validation_set.date[0].unique()[0], '%Y-%m-%d %H:%M:%S')
    total_timesteps_val = total_timesteps // (months-1)  # define time steps for training per three-month period
    f.write(f'timesteps per validation: {total_timesteps_val}\n')

    model_df = pd.DataFrame()
    # in every period the agent is retrained for a period of three month every
    for period in range(4, months + 1):
        # a dataframe to later store the period's agent's name
        period_df = pd.DataFrame()

        # splitting the period data into three month of training and one month of validation data
        train_start = initial_validation_date + pd.offsets.DateOffset(months=months_in)
        train_end = train_start + pd.offsets.DateOffset(months=3)
        test_end = train_end + pd.offsets.DateOffset(months=1)

        mask_train = (validation_set['date'] >= str(train_start)) & (validation_set['date'] <= str(train_end))
        mask_test = (validation_set['date'] > str(train_end)) & (validation_set['date'] <= str(test_end))

        val_training_set = validation_set.loc[mask_train]
        val_training_set.index = val_training_set['date'].factorize()[0]
        val_prediction_set = validation_set.loc[mask_test]
        val_prediction_set.index = val_prediction_set['date'].factorize()[0]

        # building environment and retrain the agent
        env, _ = build_env(val_training_set, env_kwargs)
        env.saving_folder = 2  # changing records folder to validation training (val_train)
        env.period = period
        # loading, retraining and saving agent
        trained_agent = load_model(method, model_name, env)
        trained_agent = train_model(trained_agent, total_timesteps=total_timesteps_val)
        model_name = save_model(trained_agent, method, period=period)

        # build new environment with the one-month validation data
        val_env, _ = build_env(val_prediction_set, env_kwargs)
        val_env.saving_folder = 1  # changing records folder to validation prediction (val_pred)
        val_env.period = period
        trained_agent = load_model(method, model_name, val_env)

        # make trading prediction for the one-month validation data
        make_prediction(trained_agent, val_env)

        # store agent name and period of agent to dataframe
        period_df['period'] = [period]
        period_df['model_name'] = [model_name]
        model_df = pd.concat([model_df, period_df], ignore_index=True)

        months_in += 1

    """
    Compare the before periodically retrained agents by evaluating their policy on the last validation month from the
    section before. Important to note is that this one-month was not used to train any model, so there is no leakage of 
    testing data.
    """

    validation_env, _ = build_env(val_prediction_set, env_kwargs)
    validation_env.saving_folder = 3
    # define metric lists for the evaluation of the before-trained models
    mean_rewards = []
    std_rewards = []
    rewards_ratio = []

    # iterate over the different models for evaluation
    for ind, agent in enumerate(model_df.model_name):
        validation_env.period = ind+4
        trained_agent = load_model(method, agent, validation_env)  # loading models from the 'models' folder
        # evaluate loaded model on last validation set
        mean_reward, std_reward = policy_evaluation(trained_agent, validation_env)

        # append the metrics to the metrics lists
        mean_rewards.append(mean_reward)
        std_rewards.append(std_reward)
        if std_reward != 0:
            rewards_ratio.append(mean_reward/std_reward)
        else:
            rewards_ratio.append(mean_reward/10e-13)

    # add metrics lists to the model dataframe
    model_df['mean_rewards'] = mean_rewards
    model_df['std_rewards'] = std_rewards
    model_df['rewards_ratio'] = rewards_ratio

    f.write(f'\n'
            f'{model_df.to_string(header=True, index=True)}'
            f'\n')

    # select the best model according to the highest reward/ standard deviation(reward) ratio
    if len(np.where(model_df['rewards_ratio'] == model_df['rewards_ratio'].max())[0]) > 1:
        model_index = int(np.where(model_df['rewards_ratio'] == model_df['rewards_ratio'].max())[0][-1])
    else:
        model_index = int(np.where(model_df['rewards_ratio'] == model_df['rewards_ratio'].max())[0])
    best_model_name = model_df['model_name'][model_index]
    print(f'The best model is: {best_model_name}')
    f.write(f'The best model is: {best_model_name}')

    after_validation_time = time.time()
    print(f'running time for validation:{round((after_validation_time - after_training_time) / 60, 0)} minutes')

elif not evaluation:
    # adjust env_kwargs dict to validation mode and load the according model from a zip file
    env_kwargs['validation'] = True
    best_model_name = MA.config.trained_model

"""
The agent is applied to the before unseen test data. After building the environment and loading the agent, it makes its
trading decisions and saves the file(s) to the according records folder.
"""
# test environment
test_env, _ = build_env(test_set, env_kwargs)
test_env.saving_folder = 4  # changing records folder to test prediction (test_pred)
print('predicting with test set')

best_agent = load_model(method, best_model_name, test_env)

make_prediction(best_agent, test_env)

f.close()
'''

elif evaluation and not run_training:
    run_eval(test_set)
    print('x')


end = time.time()
print(f'running time:{round((end - start)/60, 0)} minutes')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

# import statement
import time
import warnings


from MA.preprocessing import run_preprocess
from MA.environment_futures import build_env, show_env
from MA.agent import build_agent, train_model, save_model, make_prediction, load_model, policy_evaluation
from MA.config import DATA_PATH, env_kwargs
from MA.config import method

# time computation time
start = time.time()
warnings.filterwarnings("ignore", category=RuntimeWarning)

# fetching data
training, validation, test, processed_data = run_preprocess(DATA_PATH)
# print(training.shape)  # shape = (239670, 23)
# print(validation.shape)  # shape = (236444, 23)
# print(test.shape)  # shape = (50988, 23)

# define agent specifications
method = method

# building training environment
env, _ = build_env(training, env_kwargs)
# show_env(10, env)  # see how the environment works with actions, step, etc.

# build agent
agent = build_agent(env, method)

episodes_training = 3
episodes_validation = 3

total_timesteps = 1000

# train & save model
print('start training on training set')
for ep in range(episodes_training):
    trained_model = train_model(agent, total_timesteps=total_timesteps)

# model_name = save_model(trained_model, method)

"""
# validation environment
val_env, _ = build_env(validation, env_kwargs)
loaded_model = load_model(method, model_name, val_env)
policy_evaluation(loaded_model, val_env)

action_dict = make_prediction(loaded_model, val_env)

# after prediction, use validation set for further training
print('start training on validation set')
for ep in range(episodes_validation):
    val_trained_model = train_model(agent)
model_name_val = save_model(val_trained_model, method, validation=True)
# TODO: fine tuning of model

# test environment
test_env, _ = build_env(test, env_kwargs)
print('predicting with test set')
val_loaded_model = load_model(method, model_name_val, test_env)

policy_evaluation(val_loaded_model, test_env)

action_dict_test = make_prediction(val_trained_model, test_env, render=False)
"""

end = time.time()
print(f'running time:{round(end - start, 4)} seconds')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

# import statement
import time
import warnings

import MA.config
from MA.preprocessing import run_preprocess
from MA.environment_futures import build_env, show_env
from MA.agent import build_agent, train_model, save_model, make_prediction, load_model, policy_evaluation
from MA.config import DATA_PATH, env_kwargs
from MA.config import method, run_training

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

if run_training:
    # building training environment
    env, _ = build_env(training, env_kwargs)
    # show_env(10, env)  # see how the environment works with actions, step, etc.

    # build agent
    model = build_agent(env, method)

    episodes_training = (239670 // 500) * 2  # 2000 is approx. one month, train with each month as a random start twice
    episodes_validation = (236444 // 2000)

    total_timesteps = 1000000

    # train & save model
    print('started training on training set')
    while env.episodes <= episodes_training:
        trained_model = train_model(model, total_timesteps=total_timesteps)
    model_name = save_model(trained_model, method)

    # validation environment
    env_kwargs['validation'] = True
    """
    val_env, _ = build_env(validation, env_kwargs)
    val_env.saving_folder = 1
    loaded_model = load_model(method, model_name, val_env)
    # policy_evaluation(loaded_model, val_env)
    make_prediction(loaded_model, val_env)
    """
    # after prediction, use validation set for further training
    print('started training on validation set')
    val_env, _ = build_env(validation, env_kwargs)
    val_env.saving_folder = 2
    loaded_model = load_model(method, model_name, val_env)
    while val_env.episodes <= episodes_validation:
        val_trained_model = train_model(loaded_model, total_timesteps=total_timesteps)
    model_name_val = save_model(val_trained_model, method, validation=True)

else:
    env_kwargs['validation'] = True
    model_name_val = MA.config.trained_model

# test environment
test_env, _ = build_env(test, env_kwargs)
test_env.saving_folder = 3
print('predicting with test set')
val_loaded_model = load_model(method, model_name_val, test_env)

make_prediction(val_loaded_model, test_env)

# policy_evaluation(val_loaded_model, test_env)

end = time.time()
print(f'running time:{round(end - start, 4)} seconds')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

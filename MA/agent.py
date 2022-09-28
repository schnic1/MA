from datetime import datetime
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from MA.config import AGENT_PARAM_DICT, SAVE_MODEL_PATH


MODELS = {"a2c": A2C, "ppo": PPO}


def linear_schedule(learning_rate):
    def progress(progress_remaining):
        return progress_remaining * learning_rate
    return progress


def build_agent(env, agent):
    model_kwargs = AGENT_PARAM_DICT[f"{agent.upper()}_PARAMS"].copy()
    model_kwargs['learning_rate'] = linear_schedule(model_kwargs['learning_rate'])
    agent = Monitor(MODELS[agent]('MlpPolicy',
                                  env,
                                  verbose=0,
                                  tensorboard_log='model_data/logs',
                                  create_eval_env=True,
                                  **model_kwargs))
    return agent


def train_model(agent, total_timesteps) -> tuple:
    trained_model = agent.learn(total_timesteps=total_timesteps,
                                tb_log_name=str(agent),
                                reset_num_timesteps=False)
    return trained_model


def make_prediction(trained_model, env, render=False):

    obs = env.reset()
    done = False
    # once through the whole environment
    while not done:
        action, _state = trained_model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        if render is True:
            env.render()


def save_model(trained_model, method, path=SAVE_MODEL_PATH, validation=False, printing=False, period=0):
    if validation:
        saved_name = f'{method.upper()}_{datetime.now().strftime("%d_%m %H:%M")}'
    else:
        saved_name = f'{method.upper()}_model_{period}'

    trained_model.save(f'{path}{saved_name}')
    if printing:
        print(f'saved model as {saved_name}.zip')
    return saved_name


def load_model(method, model_name, env, path=SAVE_MODEL_PATH, printing=False):
    model = MODELS[method].load(f'{path}{model_name}', env)
    model = Monitor(model)
    if printing:
        print(f'loaded model {model_name}.zip')
    return model


def policy_evaluation(model, env):
    mean, std = evaluate_policy(model, Monitor(env), n_eval_episodes=20, render=False)
    return mean, std

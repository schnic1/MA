from stable_baselines3 import A2C, PPO, TD3, SAC, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from collections import defaultdict

from MA.config import AGENT_PARAM_DICT, SAVE_MODEL_PATH


MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

# TODO: maybe look into better Policy/improve policy?


def build_agent(env, agent):
    model_kwargs = AGENT_PARAM_DICT[f"{agent.upper()}_PARAMS"]
    agent = MODELS[agent]('MlpPolicy', env, verbose=0, tensorboard_log='logs', **model_kwargs)
    return agent


def train_model(time_steps, agent) -> tuple:
    # reset_num_timesteps=False: training continues from last training step
    trained_model = agent.learn(total_timesteps=time_steps, reset_num_timesteps=False, tb_log_name=str(agent))
    return trained_model


def make_prediction(trained_model, env, render=False):
    # to capture how often which action was taken
    action_dict = defaultdict(int)

    obs = env.reset()
    done = False
    # once through the whole environment
    while not done:
        action, _state = trained_model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        action_dict[str(action)] += 1
        if render is True:
            env.render()
    return action_dict


# TODO: does loading the model reset the whole already done training? if yes, maybe 'get_parameters' & 'set_parameters'
def save_model(trained_model, method, multiplier_training, multiplier_validation=None,  path=SAVE_MODEL_PATH):
    if multiplier_validation is not None:
        saved_name = f'{method.upper()}_{multiplier_training}x_{multiplier_validation}x'
    else:
        saved_name = f'{method.upper()}_{multiplier_training}x'
    trained_model.save(f'{path}{saved_name}')
    print(f'saved model as {saved_name}.zip')
    return saved_name


# TODO: UserWarning: Evaluation environment is not wrapped with a ``Monitor`` wrapper.
#  This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these.
#  Consider wrapping environment first with ``Monitor`` wrapper. warnings.warn(
def load_model(method, model_name, env, path=SAVE_MODEL_PATH):
    model = MODELS[method].load(f'{path}{model_name}', env)
    model = Monitor(model)
    print(f'loaded model {model_name}.zip')
    return model


def policy_evaluation(model, env):
    evaluate_policy(model, env, n_eval_episodes=5, return_episode_rewards=True)

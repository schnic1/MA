from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy


def build_agent(env):
    agent = A2C('MlpPolicy', env, verbose=0)
    return agent


def train_model(time_steps, env):
    model = build_agent(env).learn(total_timesteps=time_steps, n_eval_episodes=5)
    return model


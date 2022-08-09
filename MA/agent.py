from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy

# TODO: change Agent, make so it is input
# TODO: maybe look into better Policy?
def build_agent(env):
    agent = PPO('MlpPolicy', env, verbose=0)
    return agent


def train_model(time_steps, env):
    model = build_agent(env).learn(total_timesteps=time_steps, n_eval_episodes=6)
    return model


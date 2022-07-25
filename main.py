# import statement
import time
start = time.time()

from MA.preprocessing import run_preprocess
from MA.environment import build_env
from MA.config import DATA_PATH, env_kwargs
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


training, test, processed_data = run_preprocess(DATA_PATH)
# print(training.shape)  # 334666 datapoints !!
# until here everything is fine!!

env, _ = build_env(training, env_kwargs)
print(env.state)

"""
environment = build_environment(training, 'my_exchange')


agent = build_agent(environment)
reward = agent.train(n_steps=100, n_episodes=100)
"""

end = time.time()
print(f'running time:{round(end - start, 4)} seconds')






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

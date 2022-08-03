# import statement
import time
start = time.time()

from MA.preprocessing import run_preprocess
from MA.environment import build_env, show_env
from MA.agent import train_model
from MA.config import DATA_PATH, env_kwargs
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

training, test, processed_data = run_preprocess(DATA_PATH)
# print(f'training set shape: {training.shape} \n -----------------')  # 334666 datapoints !!

env, _ = build_env(training, env_kwargs)
# show_env(10, env)  # see how the environment works with actions, step, etc.
print(env.fetch_point_in_time())
agent = train_model(5000, env)  # 5 mio: 36225.981 seconds
print(env.fetch_point_in_time())

obs = env.state
for i in range(100):
    print(obs[:5])
    action, _state = agent.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(action, reward)

"""test_env, _ = build_env(test, env_kwargs)

obs = test_env.state

for i in range(10):
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    print(action)"""
# until here everything is fine!!

end = time.time()
print(f'running time:{round(end - start, 4)} seconds')





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

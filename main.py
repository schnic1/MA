# import statement
import time
start = time.time()

from MA.preprocessing import run_preprocess
from MA.environment import build_env, show_env
from MA.agent import train_model
from MA.config import DATA_PATH, env_kwargs
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", category=RuntimeWarning)

training, validation, test, processed_data = run_preprocess(DATA_PATH)
# print(training.shape)  # shape = (287398, 23)
# print(validation.shape)  # shape = (94354, 23)
#  print(test.shape)  # shape = (145350, 23)

env, _ = build_env(training, env_kwargs)
# show_env(10, env)  # see how the environment works with actions, step, etc.
agent = train_model(100000, env)

val_env, _ = build_env(validation, env_kwargs)
action_dict = defaultdict(int)


obs = val_env.reset()
done = False
while not done:
    action, _state = agent.predict(obs, deterministic=True)
    obs, reward, done, info = val_env.step(action)
    action_dict[str(action)] += 1


print(action_dict)
# until here everything is fine!!

end = time.time()
print(f'running time:{round(end - start, 4)} seconds')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

# import statement
from MA.preprocessing import *


import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

start = time.time()
# output dimension for RL is (1, x) so for each date we have all the features from both contracts
# if dimension needs to be changed to (2, y), some code is ready in the preprocessing.py file on the bottom
training, test, processed_data = run_preprocess("data/processed_data.pkl")
# run_preprocess("data/processed_data.pkl")

print(processed_data.head())
print(processed_data.shape)
# until here everything is fine!!
"""
environment = build_environment(training, 'my_exchange')


agent = build_agent(environment)
reward = agent.train(n_steps=100, n_episodes=100)
"""

end = time.time()
print('running time: ', (end-start), 'seconds')





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

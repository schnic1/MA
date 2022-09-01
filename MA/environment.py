import numpy as np
import pandas as pd

import gym
from gym import spaces
from gym.utils import seeding

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env


# TODO: implement changes as in "https://colab.research.google.com/drive/1tUAKi7o9NF755fUb0r3DV5HSh_xg7JZZ#scrollTo=Grjjs6tjXd46" under "Changes to Environment"


class TradingEnv(gym.Env):
    """
    Trading environment for the future contracts.
    Based on FinRL git hub repo: https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/env_stock_trading/env_stocktrading.py
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 df,
                 contract_dim,
                 action_space,
                 state_space,
                 initial_amount,
                 tech_indicators,
                 max_contracts,
                 buying_fee,  # TODO: Adjust fees to Sandro's input
                 selling_fee,
                 point_in_time=0,
                 initial=True,
                 previous_state=list,
                 print_verbosity=2,
                 ):

        self.df = df
        self.contract_dim = contract_dim

        self.tech_indicators = tech_indicators  # defined from outside
        self.initial_amount = initial_amount  # defined from outside
        self.state_space = state_space  # defined from outside, depends on # of tech indicators
        self.action_space = action_space

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.contract_dim,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))

        self.initial = initial  # defined from outside
        self.max_contracts = max_contracts  # defined from outside
        self.point_in_time = point_in_time
        self.data = self.df.loc[self.point_in_time, :]
        self.previous_state = previous_state
        self.state = self.initiate_state()

        self.terminal = False
        self.print_verbosity = print_verbosity
        self.buying_fee = buying_fee
        self.selling_fee = selling_fee

        # reward
        self.reward = 0
        self.cost = 0
        self.trades = 0
        self.episodes = 0

        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self.fetch_point_in_time()]

        self._set_seed()

    def step(self, actions) -> (np.array, float, bool, None):
        self.terminal = self.point_in_time >= len(self.df.index.unique()) - 1  # TODO: just train for one month
        if self.terminal:
            total_end_assets = self.state[0] + sum(
                np.array(self.state[1:(self.contract_dim + 1)])
                * np.array(self.state[(self.contract_dim + 1):(self.contract_dim * 2 + 1)])
                                                    )

            total_value_df = pd.DataFrame({'total value': self.asset_memory})
            total_value_df['date'] = self.date_memory
            total_value_df['returns'] = total_value_df['total value'].pct_change()

            total_rewards = (self.state[0] + sum(
                np.array(self.state[1:(self.contract_dim + 1)])
                * np.array(self.state[(self.contract_dim + 1):(self.contract_dim * 2 + 1)]))
                            - self.initial_amount)
            rewards_df = pd.DataFrame({'rewards': self.rewards_memory})
            rewards_df['date'] = self.date_memory[:-1]

            if self.episodes % self.print_verbosity == 0:
                print(f'point in time {self.point_in_time}, episode:{self.episodes}')
                print(f'total start assets: {self.asset_memory[0]:0.2f}')
                print(f'total end assets: {total_end_assets:0.2f}')
                print(f'total reward: {total_rewards:0.2f}')
                print(f'total costs: {self.cost:0.2f}')
                print(f'number of trades: {self.trades}')
                if total_value_df['returns'].std() != 0:
                    print('define sharpe ratio!')  # TODO: implement sharpe ratio
                    # sharpe_ratio =
                    # print(f'Sharpe ratio: {sharpe_ratio:0.2f}')
                print('---------------------------')

            # TODO: return statement!!! same as below but self.terminal=False again? see FinRL!!

        else:
            # define action
            actions = actions * self.max_contracts
            actions = actions.astype(int)

            step_start_asset = self.state[0] + sum(
                np.array(self.state[1:(self.contract_dim + 1)])
                * np.array(self.state[(self.contract_dim + 1):(self.contract_dim * 2 + 1)])
                                                    )
            # print(f'starting assets:{total_start_asset}')

            argsort_actions = np.argsort(actions)

            selling_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buying_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for i in selling_index:
                actions[i] = self._sell_contract(i, actions[i]) * (-1)

            for i in buying_index:
                actions[i] = self._buy_contract(i, actions[i])

            self.actions_memory.append(actions)

            # state transition from s to s+1
            self.point_in_time += 1
            self.data = self.df.loc[self.point_in_time, :]
            self.state = self.update_state()

            step_end_assets = self.state[0] + sum(
                np.array(self.state[1:(self.contract_dim + 1)])
                * np.array(self.state[(self.contract_dim + 1):(self.contract_dim * 2 + 1)])
                                                    )

            self.asset_memory.append(step_end_assets)
            self.date_memory.append(self.fetch_point_in_time())
            self.reward = step_end_assets - step_start_asset
            self.rewards_memory.append(self.reward)

            self.state = self.state.astype(np.float32)

        return self.state, self.reward, self.terminal, {}

    def _buy_contract(self, index, action) -> int:
        if self.state[index+1] > 0:
            affordable_contract = self.state[0] // self.state[index + 1]

            # update balance
            buying_num_contracts = min(affordable_contract, action)
            buying_amount = (self.state[index + 1] * buying_num_contracts * (1 + self.buying_fee))
            self.state[0] -= buying_amount

            self.state[index + self.contract_dim + 1] += buying_num_contracts

            self.cost += self.state[index + 1] * buying_num_contracts * self.buying_fee
            self.trades += 1
        else:
            buying_num_contracts = 0

        return buying_num_contracts

    def _sell_contract(self, index, action) -> int:
        if self.state[index + 1] > 0:
            if self.state[index + self.contract_dim + 1] > 0:
                selling_num_contracts = min(abs(action), self.state[index + self.contract_dim+1])
                selling_amount = (self.state[index + 1] * selling_num_contracts * (1 - self.selling_fee))

                # update balances
                self.state[0] += selling_amount
                self.state[index + self.contract_dim + 1] -= selling_num_contracts
                self.cost += (self.state[index + 1] * selling_num_contracts * self.selling_fee)
                self.trades += 1
            else:
                selling_num_contracts = 0
        else:
            selling_num_contracts = 0

        return selling_num_contracts

    def reset(self) -> np.array:
        self.state = self.initiate_state()

        if self.initial:
            self.asset_memory = [self.initial_amount]

        else:
            total_prev_asset = [self.previous_state[0]] + sum(
                np.array(self.previous_state[1:(self.contract_dim + 1)])
                * np.array(self.previous_state[(self.contract_dim + 1):(self.contract_dim * 2 + 1)])
                                                            )

            self.asset_memory = [total_prev_asset]

        self.point_in_time = 0
        self.data = self.df.loc[self.point_in_time, :]
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self.fetch_point_in_time()]

        self.episodes += 1
        self.state = self.state.astype(np.float32)

        return self.state

    def render(self, mode='human', close=False) -> None:
        return print(self.state)

    def _set_seed(self, seed=1) -> list:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def fetch_point_in_time(self) -> int:
        date = self.data.date.unique()[0]
        return date

    def initiate_state(self) -> np.array:
        # for initialising the environment & initial state
        if self.initial:
            # structure of a state in the environment (list)
            state = ([self.initial_amount]  # the balance
                     + self.data.closeprice.values.tolist()  # the closeprices of the contracts
                     + [0] * self.contract_dim  # the positions in the contracts
                     # all the technical indicators to the available contracts
                     + sum([self.data[tech_ind].values.tolist() for tech_ind in self.tech_indicators], [])
                     )
        # if using previous state
        else:
            state = ([self.previous_state[0]]
                     + self.data.closeprice.values.tolist()
                     + self.previous_state[(self.contract_dim + 1):(self.contract_dim * 2 + 1)]
                     + sum([self.data[tech_ind].values.tolist() for tech_ind in self.tech_indicators], [])
                     )
        return np.array(state)

    def update_state(self) -> np.array:
        state = ([self.state[0]]
                 + self.data.closeprice.values.tolist()
                 + list(self.state[(self.contract_dim + 1):(self.contract_dim * 2 + 1)])
                 + sum([self.data[tech_ind].values.tolist() for tech_ind in self.tech_indicators], []))
        return np.array(state)

    # TODO: parallelize environment?
    # TODO: increments episodes, needed?
    def vectorize_env(self):
        env = DummyVecEnv([lambda: self])
        obs = env.reset()

        return env, obs

    def save_asset_memory(self):
        timeline = self.date_memory
        asset_list = self.asset_memory
        df_asset_value = pd.DataFrame({'date': timeline, 'asset value': asset_list})

        return df_asset_value

    def save_action_memory(self):
        timeline = self.date_memory[:-1]
        action_list = self.actions_memory
        df_actions = pd.DataFrame({'date': timeline, 'actions': action_list})

        return df_actions


def build_env(df, specs):
    """
    build the gym environment
    :param df: (df) pandas dataframe -> training set
    :param specs: dict with specifications for the environment, can be adjusted in the config.py file
    :return: instantiated and vectorized trading environment for RL
    """
    # some more specifications for the environment depending on the dataset
    contract_dim = len(df.ticker.unique())  # number of different tickers in dataset
    # TODO: way to use tech_ind_list returned with function "create_tech_indicators"?
    tech_ind_list = df.columns.tolist()[7:]  # different indicators (they start at column index 7)

    # defining the state space of the environment
    # cash balance + (closing prices + positions per contract) + (technical indicators per contract)
    state_space = 1 + 3 * contract_dim + len(tech_ind_list) * contract_dim
    # print(f"Contract Dimension: {contract_dim}, State Space: {state_space}" \n -----)

    # add the data specific specifications to the model dict
    specs.update({"state_space": state_space,
                  "contract_dim": contract_dim,
                  "tech_indicators": tech_ind_list,
                  "action_space": contract_dim})

    # establishing and vectorizing our environment with the data and the model specification dict
    train_env = TradingEnv(df=df, **specs)
    environment_check(train_env)  # checks our environment on its implementation, raises warnings if faulty

    train_env = TradingEnv(df=df, **specs)  # newly build checked environment to start from 1 episode
    vec_train_env = train_env.vectorize_env()

    return train_env, vec_train_env


def environment_functioning(env):
    action = env.action_space.sample()
    before_state = env.state[0:5]
    print(before_state)
    env.step(action)
    print(f'action taken: {env.actions_memory[-1]}')
    after_state = env.state[0:5]
    print(after_state)
    print('reward for step:', round(env.rewards_memory[-1], 2))
    # print('total assets:', round(env.asset_memory[-1], 2))
    print('---')


def show_env(num, env):
    for n in range(num):
        print('step:', n+1)
        environment_functioning(env)


# TODO: increments episodes, maybe do it optional
def environment_check(env):
    check_env(env)

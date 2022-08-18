import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env


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
                 contract_size,
                 margins,
                 bid_ask,
                 commission,
                 margin_calls,
                 tech_indicators,
                 point_in_time=0,
                 initial=True,
                 previous_state=list,
                 print_verbosity=2):

        self.df = df
        self.contract_dim = contract_dim

        self.contract_size = contract_size
        self.margins = margins
        self.margin_calls = margin_calls
        self.bid_ask = bid_ask
        self.commission = commission
        self.initial_amount = initial_amount
        self.deposits = [0] * contract_dim

        self.max_contracts = [
            np.ceil(self.initial_amount / (self.margins[i] + self.bid_ask[i] * contract_size[i] + self.commission)) for
            i in range(self.contract_dim)]

        self.state_space = state_space
        self.action_space = action_space

        self.action_space = spaces.MultiDiscrete(self.max_contracts)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,))

        self.tech_indicators = tech_indicators

        self.initial = initial
        self.point_in_time = point_in_time
        self.data = self.df.loc[self.point_in_time, :]
        self.previous_state = previous_state

        self.state = self._initiate_state()

        self.terminal = False
        self.print_verbosity = print_verbosity

        self.asset_value = 0
        self.reward = 0
        self.costs = 0
        self.trades = 0
        self.episodes = 0

        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.pf_value_memory = []
        self.date_memory = [self.fetch_point_in_time()]

        self._set_seed()

    def step(self, actions):
        actions = np.array(actions) * np.array([2, 2]) - np.array(self.max_contracts)
        step_start_assets = self.state[0] + sum(np.array(self.state[1:(self.contract_dim + 1)]) * np.array(
            self.state[(self.contract_dim + 1):(self.contract_dim * 2 + 1)]))

        # terminal state reached
        """if:
          return 
  
        else:"""
        # take actions (buy/sell)
        argsort_actions = np.argsort(actions)

        selling_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
        buying_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

        for i in selling_index:
            actions[i] = self._sell_contract(i, actions[i]) * (-1)

        for i in buying_index:
            actions[i] = self._buy_contract(i, actions[i])

        self.actions_memory.append(actions)

        # update state, transition from s to s+1
        self.previous_state = self.state
        self.point_in_time += 1
        self.data = self.df.loc[self.point_in_time, :]
        self.state = self._update_state()

        # update deposits regarding price changes
        delta = self.state[1:(self.contract_dim + 1)] - self.previous_state[1:(self.contract_dim + 1)]
        self.deposits += delta * self.contract_size

        # margin call if necessary
        neg_depot_index = np.where(np.array(self.deposits) < 0)[0]
        for i in neg_depot_index:
            print('MARGIN CALL !')
            self.deposits[i] += self.margin_calls[i]
            self.state[0] -= self.margin_calls[i]

            # if self.state[0] negative:

        # already in t+1
        step_end_assets = self.state[0] + sum(self.deposits) + sum(
            np.array(self.state[1:(self.contract_dim + 1)]) * np.array(
                self.state[(self.contract_dim + 1):(self.contract_dim * 2 + 1)]))

        # add to memories
        self.actions_memory.append(actions)
        self.date_memory.append(self.fetch_point_in_time())
        self.pf_value_memory.append(step_end_assets)
        self.reward = step_end_assets - step_start_assets
        self.rewards_memory.append(self.reward)

        return self.state, self.reward, self.terminal, {}

    def reset(self) -> np.array:
        self.state = self._initiate_state()

        self.point_in_time = 0
        self.data = self.df.loc[self.point_in_time, :]
        self.costs = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.pf_value_memory = []
        self.date_memory = [self.fetch_point_in_time]

        self.episodes += 1

        return self.state

    # TODO: Not correct with the deposit!! Something with the ind!
    def _buy_contract(self, ind, action) -> int:
        if self.state[ind + 1] > 0:  # buying for current step closing price
            affordable_contracts = self.state[0] // (
                        self.state[ind + 1] + self.margins[ind] + self.bid_ask[ind] + self.commission)
            buying_num = min(affordable_contracts, action)

            # update balances
            buying_amount = (
                        (self.state[ind + 1] + self.margins[ind] + self.bid_ask[ind] + self.commission) * buying_num)
            self.state[0] -= buying_amount

            # update deposit
            if self.deposits[ind] == 0:
                self.deposits[ind] += self.margins[ind]  # TODO: if not or more initial margins required, add here!
            self.deposits[ind] += buying_num * self.margins[ind]

            # update position in contracts in the state
            self.state[ind + self.contract_dim + 1] += buying_num

            # update costs and trades
            self.costs += buying_num * (self.bid_ask[ind] + self.commission)
            self.trades += 1

        else:
            buying_num = 0

        return buying_num

    def _sell_contract(self, ind, actions) -> int:
        if self.state[ind + 1] > 0:  # selling for current step closing price
            if self.state[ind + self.contract_dim + 1] > 0:
                selling_num = min(abs(actions), self.state[ind + self.contract_dim + 1])

                # update balances
                # TODO: when selling, how much margin do I get back? Distinguish between last and non-last contract sold
                selling_amount = ((self.state[ind + 1] + self.margins[ind] - self.bid_ask[
                    ind] - self.commission) * selling_num)
                self.state[0] += selling_amount

                # update deposits
                self.deposits -= selling_num * self.margins[ind]
                # TODO: make sure deposits are 0 when everything is sold

                # update position in contracts in the state
                self.state[ind + self.contract_dim + 1] -= selling_num

                # update costs and trades
                self.costs += selling_amount * (self.bid_ask[ind] + self.commission)
                self.trades += 1

            else:
                selling_amount = 0
        else:
            selling_amount = 0

        return selling_amount

    def render(self, mode='human', close=False) -> None:

        return print(self.state)

    def fetch_point_in_time(self) -> int:
        time_stamp = self.data.date.unique()[0]

        return time_stamp

    def _initiate_state(self):
        if self.initial:
            state = np.array([self.initial_amount]  # cash balance
                             + self.data.closeprice.values.tolist()  # closeprices of contracts
                             + [0] * self.contract_dim  # positions in the contracts
                             + sum([self.data[tech_ind].values.tolist() for tech_ind in self.tech_indicators], []),
                             dtype=np.float32)

        else:
            state = np.array([self.previous_state[0]]
                             + self.data.closeprice.values.tolist()
                             + self.previous_state[(self.contract_dim + 1):(self.contract_dim * 2 + 1)]
                             + sum([self.data[tech_ind].values.tolist() for tech_ind in self.tech_indicators], []),
                             dtype=np.float32)

        return state

    def _update_state(self):

        state = np.array([self.state[0]]
                         + self.data.closeprice.values.tolist()
                         + list(self.state[(self.contract_dim + 1):(self.contract_dim * 2 + 1)])
                         + sum([self.data[tech_ind].values.tolist() for tech_ind in self.tech_indicators], []),
                         dtype=np.float32)

        return state

    def vectorize_env(self):
        env = DummyVecEnv([lambda: self])

        return env

    def _set_seed(self, seed=1) -> list:
        self.np_random, seed = seeding.np_random(seed)

        return [seed]


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
    state_space = 1 + 2 * contract_dim + len(tech_ind_list) * contract_dim
    # print(f"Contract Dimension: {contract_dim}, State Space: {state_space}" \n -----)

    # add the data specific specifications to the model dict
    specs.update({"state_space": state_space,
                  "contract_dim": contract_dim,
                  "tech_indicators": tech_ind_list,
                  "action_space": contract_dim})

    # establishing and vectorizing our environment with the data and the model specification dict
    train_env = TradingEnv(df=df, **specs)
    environment_check(train_env)

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

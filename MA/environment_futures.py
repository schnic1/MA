import numpy as np

import gym
import pandas as pd
from gym import spaces

from datetime import datetime

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
                 tech_indicators,
                 validation,
                 point_in_time=0,
                 initial=True,
                 previous_state=list,
                 print_verbosity=1):

        self.df = df
        self.initial = initial
        self.initial_step = point_in_time
        self.point_in_time = point_in_time
        self.data = self.df.loc[self.point_in_time, :]
        self.contract_dim = contract_dim

        self.contract_size = contract_size
        self.margins = margins
        self.bid_ask = bid_ask
        self.commission = commission
        self.initial_amount = initial_amount
        # self.deposits = [0] * contract_dim
        # self.contract_positions = [0] * contract_dim
        self.contract_positions_initial = [2] * contract_dim
        self.contract_positions = self.contract_positions_initial.copy()
        self.initial_deposits = [float(36000),  # self.contract_positions[0]*self.margins[0],
                                 float(10000)]  # self.contract_positions[1]*self.margins[1]]
        self.deposits = self.initial_deposits.copy()

        self.max_contracts = [
            np.ceil(self.initial_amount / (self.margins[i] + self.bid_ask[i] * contract_size[i] + self.commission) + 1)
            for i in range(self.contract_dim)]
        self.action_contracts = [2 * max_cont for max_cont in self.max_contracts]

        self.state_space = state_space
        self.action_space = action_space

        self.action_space = spaces.MultiDiscrete(self.action_contracts)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,))

        self.tech_indicators = tech_indicators
        self.last_non_zero_prices = np.array([0] * contract_dim)

        self.previous_state = previous_state
        self.state = self._initiate_state()

        self.terminal = False
        self.margin_call = False
        self.default = False
        self.validation = validation
        self.print_verbosity = print_verbosity

        self.reward = 0
        self.costs = 0
        self.trades = 0

        self.episodes = 0

        # initialising memories
        self.cash_memory = [self.initial_amount]
        self.deposits_memory = [self.initial_deposits]
        self.positions_memory = [self.contract_positions_initial]
        self.date_memory = [self._fetch_point_in_time()]

        self.cost_memory = []
        self.total_costs_memory = []
        self.rewards_memory = []
        self.total_reward_memory = []
        self.actions_memory = []
        self.margin_call_memory = []
        self.pf_value_memory = []
        self.PnL_memory = []

    def step(self, actions):
        # uncomment print statements to see functioning
        print(self._fetch_point_in_time())
        print('at step start: cash:', self.state[0],
              '| positions:', self.contract_positions,
              '| deposit:', self.deposits)

        int_actions = np.array(actions) - np.array(self.max_contracts)
        actions = np.array(actions) - np.array(self.max_contracts)
        step_start_assets = (self.state[0] + sum(self.deposits))

        begin_costs = self.costs

        # smaller volume is bought first
        buying_index = np.where(actions > 0)[0]

        selling_index = np.where(actions < 0)[0]

        # first buy, then sell
        if (actions[0] > actions[1]) and (actions[1] > 0):
            for i in buying_index[::-1]:
                actions[i] = self._buy_contract(i, actions[i])
        else:
            for i in buying_index:
                actions[i] = self._buy_contract(i, actions[i])

        for i in selling_index:
            actions[i] = self._sell_contract(i, actions[i]) * (-1)

        # update state, transition from s to s+1
        self.previous_state = self.state
        self.point_in_time += 1
        self.data = self.df.loc[self.point_in_time, :]
        self.state = self._update_state()

        # update deposits regarding price changes
        past_prices = np.array(self.previous_state[1:(self.contract_dim + 1)])
        upcoming_prices = np.array(self.state[1:(self.contract_dim + 1)])

        delta = upcoming_prices - past_prices

        if float(0) in upcoming_prices:
            index = np.where(upcoming_prices == 0)[0]
            for ind in index:
                delta[ind] = 0
                if self.last_non_zero_prices[ind] == 0:
                    self.last_non_zero_prices[ind] = past_prices[ind]

        if float(0) in past_prices:
            # print('past price is 0')
            index = np.where(past_prices == 0)[0]
            for ind in index:
                if self.last_non_zero_prices[ind] != 0 and upcoming_prices[ind] != 0:
                    delta[ind] = upcoming_prices[ind] - self.last_non_zero_prices[ind]
                    self.last_non_zero_prices[ind] = 0
                else:
                    delta[ind] = 0
        after_trade_before_step_deposit = self.deposits
        self.deposits += delta * np.array(self.contract_size) * np.array(self.contract_positions)
        step_deposit = self.deposits.copy()
        step_pos = self.contract_positions.copy()

        end_costs = self.costs
        step_costs = end_costs - begin_costs

        margin_calls = [0] * 2
        neg_depot_index = np.where(np.array(self.deposits) < np.array(self.contract_positions) * np.array(self.margins))[0]
        for i in neg_depot_index:
            margin_call = self.contract_positions[i] * self.margins[i] - self.deposits[i]
            self.deposits[i] += margin_call
            self.state[0] -= margin_call
            margin_calls[i] = margin_call

            # if cash (self.state[0]) is negative:
            if self.state[0] < 0:
                self.default = True

        # already in t+1
        step_end_assets = (self.state[0] + sum(self.deposits))

        print('intended trades:', int_actions)
        print('effective trades:', actions)
        print('prices before trade:', np.array(self.previous_state[1:(self.contract_dim + 1)]))
        print('margin calls:', margin_calls)
        print('prices after trade:', np.array(self.state[1:(self.contract_dim + 1)]))
        print('delta prices:', delta)
        print('after time step(end) cash:', self.state[0],
              '| positions:', self.contract_positions,
              '| deposit:', self.deposits)
        print('total costs:', self.costs, ' trades done:', self.trades)
        print('step start assets:', step_start_assets,
              'step_end_assets:', step_end_assets,
              'asset delta:', step_end_assets-step_start_assets,
              '\n', '------')

        # add everything to memories
        self.actions_memory.append(actions)
        self.positions_memory.append(step_pos)
        self.deposits_memory.append(list(step_deposit))
        self.date_memory.append(self._fetch_point_in_time())
        self.pf_value_memory.append(step_end_assets)
        self.cash_memory.append(self.state[0])
        self.cost_memory.append(step_costs)
        self.total_costs_memory.append(end_costs)
        self.margin_call_memory.append(margin_calls)

        # define rewards
        asset_delta = step_end_assets - step_start_assets
        self.PnL_memory.append(asset_delta)

        step_penalties = 0
        if self.default:
            step_penalties += 10000

        # TODO: more sophisticated rewards
        self.reward = asset_delta - step_penalties
        self.rewards_memory.append(self.reward)
        self.total_reward_memory.append(self.total_reward_memory[-1]+self.reward)
        # print(self.reward)

        # terminal state is reached when either at end of data or reached 3 month
        if self.point_in_time >= len(self.df.index.unique()) - 1:
            self.terminal = True
        else:
            starting_point_in_time = datetime.strptime(str(self.date_memory[0]), '%Y-%m-%d %H:%M:%S')
            current_point_in_time = datetime.strptime(str(self.date_memory[-1]), '%Y-%m-%d %H:%M:%S')

            self.terminal = (current_point_in_time - starting_point_in_time).days >= 90

        if self.default:
            self.terminal = True

        # terminal state reached
        if self.terminal:
            # create and export dataframes to csv

            episode_data = {'date': self.date_memory,
                            'cash': self.cash_memory,
                            'actions': self.actions_memory,
                            'positions': self.positions_memory,
                            'deposits': self.deposits_memory,
                            'margin calls': self.margin_call_memory,
                            'PF value': self.pf_value_memory,
                            'costs': self.cost_memory,
                            'total costs': self.total_costs_memory,
                            'rewards': self.rewards_memory,
                            'total reward': self.total_reward_memory,
                            'PnL': self.PnL_memory
                            }

            episode_df = pd.DataFrame(data=episode_data)
            episode_df['returns'] = episode_df['PF value'].pct_change(1)
            episode_df.to_csv(f'episode_data/episode_{self.episodes}_{datetime.now().strftime("%d_%m %H_%M")}.csv')

            if self.episodes % self.print_verbosity == 0:
                print(f'point in time: {self._fetch_point_in_time()}, episode: {self.episodes}')
                print(f'steps done: {self.point_in_time - self.initial_step}')
                print(f'beginning assets: {self.pf_value_memory[0]:0.2f}')
                print(f'final assets: {step_end_assets:0.2f}')
                print(f'total rewards: {sum(self.rewards_memory):0.2f}')
                print(f'total PnL: {sum(self.PnL_memory):0.2f}')
                print(f'total costs: {self.costs:0.2f}')
                print(f'total trades: {self.trades:0.2f}')
                print('----')

            self.reset()

        return self.state, self.reward, self.terminal, {}

    def reset(self) -> np.array:
        if not self.initial:
            self.state = self._initiate_state()
            self.initial = True

        else:
            # for validation start at beginning then roll one month after each episode
            if self.validation:
                self.point_in_time = 0 + 2000 * self.episodes  # if started from beginning or specific point in time

            # for training start randomly, after each episode, chose random starting point
            else:
                self.point_in_time = np.random.randint(len(self.df.index.unique())-1)

            self.initial_step = self.point_in_time
            self.data = self.df.loc[self.point_in_time, :]

            self.contract_positions = self.contract_positions_initial.copy()
            self.deposits = self.initial_deposits.copy()

            self.state = self._initiate_state()

            pf_value_start = (self.state[0]
                              + sum(np.array(self.state[1:(self.contract_dim + 1)]) * self.contract_positions)
                              + sum(self.deposits)
                              )

            self.costs = 0
            self.trades = 0
            self.reward = 0
            self.terminal = False
            self.default = False

            self.cash_memory = [self.initial_amount]
            self.pf_value_memory = [pf_value_start]
            self.deposits_memory = [self.initial_deposits]
            self.positions_memory = [self.contract_positions_initial]
            self.date_memory = [self._fetch_point_in_time()]

            self.total_costs_memory = [0]
            self.cost_memory = [0]
            self.total_reward_memory = [0]
            self.rewards_memory = [0]
            self.actions_memory = [[0, 0]]
            self.PnL_memory = [0]
            self.margin_call_memory = [[0, 0]]

        self.episodes += 1

        return self.state

    def _buy_contract(self, ind, action) -> int:
        # check whether price is available, otherwise no trade
        if self.state[ind + 1] > float(0):  # buying for current step closing price
            # total money taken from cash per contract
            money_per_contract = (self.margins[ind]
                                  + self.bid_ask[ind] * self.contract_size[ind]
                                  + self.commission)
            # total cost per one contract
            costs_per_contract = (self.bid_ask[ind] * self.contract_size[ind] + self.commission)

            affordable_contracts = self.state[0] // money_per_contract

            buying_num = min(affordable_contracts, action)

            buying_amount = money_per_contract * buying_num

            self.state[0] -= buying_amount

            # put margins into deposit
            self.deposits[ind] += buying_num * self.margins[ind]

            # update position in contracts in the state
            self.contract_positions[ind] += buying_num

            # update costs and trades
            self.costs += buying_num * costs_per_contract
            if buying_num != 0:
                self.trades += 1

        else:
            buying_num = 0

        return buying_num

    def _sell_contract(self, ind, actions) -> int:
        # check whether price is available, otherwise no trade
        if self.state[ind + 1] > 0:  # selling for current step closing price
            if self.contract_positions[ind] > float(0):
                selling_num = min(abs(actions), self.contract_positions[ind])
                # total money getting from cash per contract
                money_per_contract = (self.margins[ind]
                                      - self.bid_ask[ind] * self.contract_size[ind]
                                      - self.commission)
                # total cost per one contract
                costs_per_contract = (self.bid_ask[ind] * self.contract_size[ind] + self.commission)

                # update balances
                # TODO: when selling, how much margin do I get back? Distinguish between last and non-last contract sold
                selling_amount = money_per_contract * selling_num
                self.state[0] += selling_amount

                # update position in contracts
                self.contract_positions[ind] -= selling_num

                # update deposits and return initial margin if position is 0
                self.deposits[ind] -= selling_num * self.margins[ind]
                if self.contract_positions[ind] == float(0):
                    self.state[0] += self.deposits[ind]
                    self.deposits[ind] = 0

                # update costs and trades
                self.costs += selling_num * costs_per_contract
                self.trades += 1

            else:
                selling_num = 0
        else:
            selling_num = 0

        return selling_num

    def render(self, mode='human', close=False) -> None:

        return print(self.state)

    def _fetch_point_in_time(self) -> int:
        time_stamp = self.data.date.unique()[0]

        return time_stamp

    def _initiate_state(self):
        state = np.array([self.initial_amount]  # cash balance
                         + self.data.closeprice.values.tolist()  # closeprices of contracts
                         + self.contract_positions_initial  # positions in the contracts
                         + self.initial_deposits  # balance on deposits
                         + sum([self.data[tech_ind].values.tolist() for tech_ind in self.tech_indicators], []),
                         dtype=np.float32)
        return state

    def _update_state(self):

        state = np.array([self.state[0]]
                         + self.data.closeprice.values.tolist()
                         + list(self.contract_positions)
                         + list(self.deposits)
                         + sum([self.data[tech_ind].values.tolist() for tech_ind in self.tech_indicators], []),
                         dtype=np.float32)

        return state

    def vectorize_env(self):
        env = DummyVecEnv([lambda: self])

        return env


def environment_functioning(env):
    action = env.action_space.sample()
    before_state = env.state[0:5]
    print(before_state)
    env.step(action)
    print(f'action taken: {env.actions_memory[-1]}')
    after_state = env.state[0:5]
    print(after_state)
    print('reward for step:', round(env.rewards_memory[-1], 2))
    print('---')


def show_env(num, env):
    for n in range(num):
        print('step:', n + 1)
        environment_functioning(env)


# TODO: increments episodes, maybe do it optional
def environment_check(env):
    check_env(env)


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
    # cash balance + (closing prices + positions + deposits per contract) + (technical indicators per contract)
    state_space = 1 + 3 * contract_dim + len(tech_ind_list) * contract_dim

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
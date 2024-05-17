import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from finrl.meta.preprocessor.preprocessors import data_split
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import SAC
from stable_baselines3 import A2C
from stable_baselines3 import TD3
from finrl.main import check_and_make_directories
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.config import INDICATORS
from finrl.plot import backtest_stats

# test
DIR = '../Data/data2/'

TRADE_START_DATE = '2016-01-07'
TRADE_END_DATE = '2017-01-07'
DATA_NAME = 'full-DOW29' 

# TRADE_START_DATE = '2017-01-09'
# TRADE_END_DATE = '2018-01-09'
# DATA_NAME = 'full-HS63'  
# TIC_DIR = '../MOE/selectedAssets/full-HS63/test3/'

# TRADE_START_DATE = '2023-11-12'
# TRADE_END_DATE = '2024-04-30'
# DATA_NAME = 'full-CRYPTO33'
# TIC_DIR = '../MOE/selectedAssets/full-CRYPTO33/test2/'


# TRADE_START_DATE = '2021-01-05'
# TRADE_END_DATE = '2022-01-06'
# DATA_NAME = 'full-HK73' 
# TIC_DIR = '../MOE/selectedAssets/full-HK73/test6/'


# TRADE_START_DATE = '2019-01-04'
# TRADE_END_DATE = '2020-01-04'
# DATA_NAME = 'full-NYSE62'  
# TIC_DIR = '../MOE/selectedAssets/full-NYSE62/test3/'


# TRADE_START_DATE = '2023-01-06'
# TRADE_END_DATE = '2024-01-06'
# DATA_NAME = 'full-FTSE75' 
# TIC_DIR = '../MOE/selectedAssets/full-FTSE75/test10/'


FILE_PATH = DIR + DATA_NAME + '.csv'



class Test_Single_Expert_obj:

    def __init__(self, environment, model, stocksDimension, hmax=100):
        self.stocksDim = stocksDimension
        self.hamx =hmax
        self.last_obs = None
        self.environment = environment
        self.model = model
        pass

    def init_predict(self):

        """make a prediction and get results"""
        self.test_env, self.test_obs = self.environment.get_sb_env()
        self.account_memory = None  # This help avoid unnecessary list creation
        self.actions_memory = None  # optimize memory consumption
        # state_memory=[] #add memory pool to store states
        self.test_env.reset()
        self.max_steps = len(self.environment.df.index.unique()) - 1


    def sparse_action(self, env, action, stocks_num):
        """
        Sparse action function that generates a sparse action based on the given environment, action, and number of stocks.

        :param env: (object) Environment object
        :param action: (array[0][stocks_num]) Action array
        :param stocks_num: (int) Number of stocks
        :return: (array[0][stocks_num]) Sparse action array
        """
        # Get the currently held quantity of stocks in the env
        heldQuantity = env.state[stocks_num + 1:2 * stocks_num + 1]
        newAction = np.array(heldQuantity)/-self.hamx

        # Find the index of the maximum value in the action array
        maxActionIndex = np.argmax(action[0])
        if action[0][int(maxActionIndex)] >= 0:
            newAction[int(maxActionIndex)] = 1000000.0
        return [newAction]

    def DRL_prediction(self, i):

        action, _states = self.model.predict(self.test_obs, deterministic=True)

        action = self.sparse_action(self.environment, action, 29)  
        return action, self.test_obs[0][0]


if __name__ == '__main__':
    # We will use unseen, post-deadline data for testing
    parser = argparse.ArgumentParser(description='Description of program')
    parser.add_argument('--start_date', default=TRADE_START_DATE,
                        help='Trade start date (default: {})'.format(TRADE_START_DATE))
    parser.add_argument('--end_date', default=TRADE_END_DATE,
                        help='Trade end date (default: {})'.format(TRADE_END_DATE))
    parser.add_argument('--data_file', default=FILE_PATH, help='Trade data file')

    args = parser.parse_args()
    TRADE_START_DATE = args.start_date
    TRADE_END_DATE = args.end_date

    processed_full = pd.read_csv(args.data_file)
    processed_full['date'] = pd.to_datetime(processed_full['date']).dt.strftime('%Y-%m-%d') 
    total_trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)


    stock_dimension = len(total_trade.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    # sample
    stock_dimension_sample = 29
    state_space_sample = 1 + 2 * stock_dimension_sample + len(INDICATORS) * stock_dimension_sample
    print(f"Stock Dimension Sample: {stock_dimension_sample}, State Space Sample: {state_space_sample}")
    buy_cost_list_sample = sell_cost_list_sample = [0.001] * stock_dimension_sample
    num_stock_shares_sample = [0] * stock_dimension_sample

    total_tics = list(total_trade.tic.unique())
    trades = []
    if stock_dimension > 29:
        sample_num = math.ceil(stock_dimension / 29)
        for i in range(sample_num):
            if DATA_NAME == 'full-DOW29':
                random_tic = random.sample(total_tics, 29)
            else:
                temp_df = pd.read_csv(TIC_DIR + str(i) + '_choose.csv')
                random_tic = list(temp_df.tic.unique())

            # print(i, random_tic)
            df = total_trade[total_trade['tic'].isin(random_tic)]
            trades.append(df)

    # please do not change initial_amount
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    env_kwargs_sample = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares_sample,
        "buy_cost_pct": buy_cost_list_sample,
        "sell_cost_pct": sell_cost_list_sample,
        "state_space": state_space_sample,
        "stock_dim": stock_dimension_sample,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension_sample,
        "reward_scaling": 1e-4
    }

    check_and_make_directories([TRAINED_MODEL_DIR])

    # Environment
    e_total_trade_gym = StockTradingEnv(df=total_trade, **env_kwargs)


    # agent
    agent = DRLAgent(env=e_total_trade_gym)

    # test nine agents :
    trained_agent = PPO.load('./trained_models/' + 'ppoReal1')

    trade_total = Test_Single_Expert_obj(environment=e_total_trade_gym,
                                         model=trained_agent, stocksDimension=stock_dimension)

    trade_total.init_predict()

    if stock_dimension > 29:
        traders = []
        n_trade = math.ceil(stock_dimension / 29)
        for i in range(n_trade):
            e_trade_gym = StockTradingEnv(df=trades[i], **env_kwargs_sample)
            trader = Test_Single_Expert_obj(environment=e_trade_gym, model=trained_agent, stocksDimension=stock_dimension)
            traders.append(trader)

        for t in traders:
            t.init_predict()


        for i in range(len(e_total_trade_gym.df.index.unique())):
            max_cash = 0
            max_action = None
            max_tic = None

            for j in range(len(traders)):
                action, cash = traders[j].DRL_prediction(i)
                if cash > max_cash:
                    max_cash = cash
                    max_action = action
                    max_tic = list(trades[j].tic.unique())
                traders[j].test_obs, _, _, _ = traders[j].test_env.step(action)

            final_action = [np.array([0 for _ in range(stock_dimension)])]
            index_list = []
            for k in range(len(max_tic)):
                index = total_tics.index(max_tic[k])
                final_action[0][index] = max_action[0][k]
           
            trade_total.test_obs, _, dones, _ = trade_total.test_env.step(final_action)
           
            if i == trade_total.max_steps - 1:  # more descriptive condition for early termination to clarify the logic
                account_memory = trade_total.test_env.env_method(method_name="save_asset_memory")
                actions_memory = trade_total.test_env.env_method(method_name="save_action_memory")
            if dones[0]:
                print("hit end!")
                print(account_memory[0])
                df_result_agent = account_memory[0]
                break
    else:
        trade_total.init_predict()
        for i in range(len(e_total_trade_gym.df.index.unique())):
            final_action, _ = trade_total.DRL_prediction(i)
            trade_total.ppo_switch_obs, _, dones, _ = trade_total.test_env.step(final_action)
            if i == trade_total.max_steps -1:
                account_memory = trade_total.test_env.env_method(method_name="save_asset_memory")
                actions_memory = trade_total.test_env.env_method(method_name="save_action_memory")
            if dones[0]:
                print("hit end!")
                print(account_memory[0])
                df_result_agent = account_memory[0]
                break

    print("==============Get Backtest Results===========")
    perf_stats_all = backtest_stats(account_value=df_result_agent)

    """Plotting"""
    # plt.rcParams["figure.figsize"] = (15, 5)
    # plt.figure()

    # df_result_ppo.plot()
    print(df_result_agent.iloc[-1])

    # print(df_result_agent.iloc[-1])
    # df_result_agent.to_csv("./singlefinalResult/" + DATA_NAME[5:] + '/' + "ppoMax1.csv", index=False)

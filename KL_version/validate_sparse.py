import os

import numpy as np
import pandas as pd
import argparse
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.main import check_and_make_directories
from finrl.config import TRAINED_MODEL_DIR
from finrl.config import INDICATORS
from finrl.plot import backtest_stats
from stable_baselines3 import DDPG, SAC, TD3

from KL_version.my_a2c import MyA2C
from my_ppo import MyPPO
from my_sac import MySAC


# validate
DIR = '../Data/data/'

TRADE_START_DATE = '2014-01-05'
TRADE_END_DATE = '2015-01-06'
DATA_NAME = 'DOW29/full-DOW29_Real'

# TRADE_START_DATE = '2015-01-06'
# TRADE_END_DATE = '2016-01-07'
# DATA_NAME = 'HS29/full-HS29_Real'

# TRADE_START_DATE = '2021-11-10'
# TRADE_END_DATE = '2022-11-11'
# DATA_NAME = 'CRYPTO29/full-CRYPTO29_Real'

# TRADE_START_DATE = '2019-01-03'
# TRADE_END_DATE = '2020-01-04'
# DATA_NAME = 'HK29/full-HK29_Real'

# TRADE_START_DATE = '2017-01-02'
# TRADE_END_DATE = '2018-01-03'
# DATA_NAME = 'NYSE29/full-NYSE29_Real'

# TRADE_START_DATE = '2021-01-04'
# TRADE_END_DATE = '2022-01-05'
# DATA_NAME = 'FTSE29/full-FTSE29_Real'

FILE_PATH = DIR + DATA_NAME + '.csv'


def sparse_action(env, action, stocks_num):
    """
    Sparse action function that generates a sparse action based on the given environment, action, and number of stocks.

    :param env: (object) Environment object
    :param action: (array[0][stocks_num]) Action array
    :param stocks_num: (int) Number of stocks
    :return: (array[0][stocks_num]) Sparse action array
    """
    # Get the currently held quantity of stocks in the env
    heldQuantity = env.state[stocks_num + 1:2 * stocks_num + 1]
    newAction = np.array(heldQuantity) / -100  # hmax

    # Find the index of the maximum value in the action array
    maxActionIndex = np.argmax(action[0])
    if action[0][int(maxActionIndex)] >= 0:
        newAction[int(maxActionIndex)] = 1000000.0
    return [newAction]

def DRL_prediction(model, environment, deterministic=True):
    """make a prediction and get results"""
    test_env, test_obs = environment.get_sb_env()
    account_memory = None  # This help avoid unnecessary list creation
    actions_memory = None  # optimize memory consumption
    # state_memory=[] #add memory pool to store states

    test_env.reset()
    max_steps = len(environment.df.index.unique()) - 1

    for i in range(len(environment.df.index.unique())):
        action, _states = model.predict(test_obs, deterministic=deterministic)
        # account_memory = test_env.env_method(method_name="save_asset_memory")
        # actions_memory = test_env.env_method(method_name="save_action_memory")

        # Sparse the chooseAction (invest in only one stock)
        action = sparse_action(environment, action, stocks_num=29) 

        test_obs, rewards, dones, info = test_env.step(action)


        if (
                i == max_steps - 1
        ):  # more descriptive condition for early termination to clarify the logic
            account_memory = test_env.env_method(method_name="save_asset_memory")
            actions_memory = test_env.env_method(method_name="save_action_memory")
        # add current state to state memory
        # state_memory=test_env.env_method(method_name="save_state_memory")

        if dones[0]:
            print("hit end!")
            break
    return account_memory[0], actions_memory[0]

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
    trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)

    stock_dimension = len(trade.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

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

    # Environment
    e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)

    # need to validate:
    # methods = ['ppoReal1', 'ppoReal2', 'ppoReal3', 'ppoReal4']
    # methods = ['ppoMax1', 'ppoMax2', 'ppoMax3', 'ppoMax4']
    # methods = ['ppoMin1', 'ppoMin2', 'ppoMin3', 'ppoMin4']
    # methods = ['ppoMean1', 'ppoMean2', 'ppoMean3', 'ppoMean4']
    # methods = ['ppoEma1', 'ppoEma2', 'ppoEma3', 'ppoEma4']

    # methods = ['a2cMax1', 'a2cMax2', 'a2cMax3', 'a2cMax4',
    #            'a2cMin1', 'a2cMin2', 'a2cMin3', 'a2cMin4',
    #            'a2cMean1', 'a2cMean2', 'a2cMean3', 'a2cMean4',
    #            'a2cEma1', 'a2cEma2', 'a2cEma3', 'a2cEma4']

    trained_model = MyPPO.load(TRAINED_MODEL_DIR + '/' + 'ppoReal1')


    # Backtesting
    df_result_ppo, df_actions_ppo = DRL_prediction(model=trained_model, environment=e_trade_gym)

    print("==============Get Backtest Results===========")
    perf_stats_all = backtest_stats(account_value=df_result_ppo)

    # print(df_result_ppo)
    print(df_result_ppo.iloc[-1])

    # DATA_NAME = DATA_NAME.split('/')[0]
    # save_dir = "./validateSparseResults/" + str(DATA_NAME) + "/"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # save
    # df_result_ppo.to_csv(save_dir + str(method) + '_results.csv', index=False)

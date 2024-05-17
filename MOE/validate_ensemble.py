import math
import os
import random
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

import torch
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import PPO, A2C
from finrl.main import check_and_make_directories
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.plot import backtest_stats

from KL_version.my_a2c import MyA2C
from KL_version.my_ppo import MyPPO
from MOE.MoE import MoE
from MOE.ensemble_final_obj import Ensemble_Final_Obj

# sample ten to get the best

DIR = '../Data/data2/'

# TRADE_START_DATE = '2015-01-06'
# TRADE_END_DATE = '2016-01-07'
# DATA_NAME = 'full-DOW29'

TRADE_START_DATE = '2016-01-07'
TRADE_END_DATE = '2017-01-07'
DATA_NAME = 'full-HS63'
#
# TRADE_START_DATE = '2022-11-11'
# TRADE_END_DATE = '2023-11-12'
# DATA_NAME = 'full-CRYPTO33'
#
# TRADE_START_DATE = '2020-01-06'
# TRADE_END_DATE = '2021-01-05'
# DATA_NAME = 'full-HK73'
#
# TRADE_START_DATE = '2018-01-03'
# TRADE_END_DATE = '2019-01-04'
# DATA_NAME = 'full-NYSE62'
#
# TRADE_START_DATE = '2022-01-05'
# TRADE_END_DATE = '2023-01-05'
# DATA_NAME = 'full-FTSE75'


FILE_PATH = DIR + DATA_NAME + '.csv'

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
        sample_num = math.ceil(stock_dimension/29)
        for i in range(sample_num):  
            random_tic = random.sample(total_tics, 29)
            print(i, random_tic)
            df = total_trade[total_trade['tic'].isin(random_tic)]
            trades.append(df)

    # save
    # save_dir = './chooseStock/' + DATA_NAME + '/test10/'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #     # save
    #     for i in range(len(trades)):
    #         trades[i].to_csv(save_dir + str(i) + '_choose.csv', index=False)

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

    # agents
    ppoReal1 = MyPPO.load('../KL_version/trained_models/' + 'ppoReal1')
    a2cEma1 = MyA2C.load('../KL_version/trained_models/' + 'a2cEma1')
    a2cMean4 = MyA2C.load('../KL_version/trained_models/' + 'a2cMean4')
    ppoMin3 = MyPPO.load('../KL_version/trained_models/' + 'ppoMin3')
    a2cEma4 = MyA2C.load('../KL_version/trained_models/' + 'a2cEma4')
    ppoReal4 = MyPPO.load('../KL_version/trained_models/' + 'ppoReal4')
    a2cMax2 = MyA2C.load('../KL_version/trained_models/' + 'a2cMax2')
    ppoMax3 = MyPPO.load('../KL_version/trained_models/' + 'ppoMax3')
    ppoMax1 = MyPPO.load('../KL_version/trained_models/' + 'ppoMax1')

    # load moe
    model_name = 'trained_moe_model.pth'
    moe_model = MoE(
        trained_experts=[ppoReal1, a2cEma1, a2cMean4, ppoMin3, a2cEma4, ppoReal4, a2cMax2, ppoMax3, ppoMax1])
    checkpoint = torch.load('./trained_models/' + model_name)
    moe_model.load_state_dict(checkpoint['model_state_dict'])
    moe_model.eval()

    # Environment
    e_total_trade_gym_switch = StockTradingEnv(df=total_trade, **env_kwargs)
    ppo_switch_total = Ensemble_Final_Obj(environment=[e_total_trade_gym_switch], model_moe=moe_model,
                                        switchWindows=[6, 9, 10], alpha=0.45,
                                        hmax=env_kwargs['hmax'],
                                        stocksDimension=stock_dimension)
    ppo_switch_total.init_predict()

    if stock_dimension > 29:
        ppo_switchs = []
        n_switch = math.ceil(stock_dimension / 29)
        for i in range(n_switch):
            e_trade_gym_switch = StockTradingEnv(df=trades[i], **env_kwargs_sample)
            ppo_switch = Ensemble_Final_Obj(environment=[e_trade_gym_switch], model_moe=moe_model, switchWindows=[6, 9, 10], alpha=0.45, hmax=env_kwargs['hmax'],
                                          stocksDimension=29)
            ppo_switchs.append(ppo_switch)

        for ppo_switch in ppo_switchs:
            ppo_switch.init_predict()

        for i in range(len(e_total_trade_gym_switch.df.index.unique())):
            max_cash = 0
            max_action = None
            max_tic = None

            for j in range(len(ppo_switchs)):
                action, cash = ppo_switchs[j].DRL_prediction(i)
                if cash > max_cash:
                    max_cash = cash
                    max_action = action
                    max_tic = list(trades[j].tic.unique())
                ppo_switchs[j].ppo_switch_obs, _, _, _ = ppo_switchs[j].ppo_switch_env.step(action)

            final_action = [np.array([0 for _ in range(stock_dimension)])]
            index_list = []
            for k in range(len(max_tic)):
                index = total_tics.index(max_tic[k])
                final_action[0][index] = max_action[0][k]
          
            ppo_switch_total.ppo_switch_obs, _, dones, _ = ppo_switch_total.ppo_switch_env.step(final_action)
            # print(ppo_switch_total.ppo_switch_obs)
            if i == ppo_switch_total.max_steps - 1:  # more descriptive condition for early termination to clarify the logic
                account_memory = ppo_switch_total.ppo_switch_env.env_method(method_name="save_asset_memory")
                actions_memory = ppo_switch_total.ppo_switch_env.env_method(method_name="save_action_memory")
            if dones[0]:
                print("hit end!")
                print(account_memory[0])
                df_result_ppo = account_memory[0]
                break
    else:
        ppo_switch_total.init_predict()
        for i in range(len(e_total_trade_gym_switch.df.index.unique())):
            final_action, _ = ppo_switch_total.DRL_prediction(i)
            ppo_switch_total.ppo_switch_obs, _, dones, _ = ppo_switch_total.ppo_switch_env.step(final_action)
            if i == ppo_switch_total.max_steps -1 :
                account_memory = ppo_switch_total.ppo_switch_env.env_method(method_name="save_asset_memory")
                actions_memory = ppo_switch_total.ppo_switch_env.env_method(method_name="save_action_memory")
            if dones[0]:
                print("hit end!")
                print(account_memory[0])
                df_result_ppo = account_memory[0]
                break

    # print("==============Get Backtest Results===========")
    perf_stats_all = backtest_stats(account_value=df_result_ppo)

  

    # print(df_result_ppo.iloc[0])
    print('------', df_result_ppo.iloc[-1])

    
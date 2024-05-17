import datetime
import os

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
from torch import optim

from KL_version.my_a2c import MyA2C
from KL_version.my_ppo import MyPPO
from MOE.MoE import MoE
from MOE.train_moe_esemble import PPO_Switch

# Contestants are welcome to split the data in their own way for model tuning
DIR = '../Data/data/'
TRAIN_START_DATE = 'TRAIN_START_DATE'
TRAIN_END_DATE = 'TRAIN_END_DATE'
DATA_NAME = 'forMoE'
FILE_PATH = DIR + DATA_NAME + '.csv'

if __name__ == '__main__':
    # We will use unseen, post-deadline data for testing
    parser = argparse.ArgumentParser(description='Description of program')
    parser.add_argument('--start_date', default=TRAIN_START_DATE,
                        help='Trade start date (default: {})'.format(TRAIN_START_DATE))
    parser.add_argument('--end_date', default=TRAIN_END_DATE,
                        help='Trade end date (default: {})'.format(TRAIN_END_DATE))
    parser.add_argument('--data_file', default=FILE_PATH, help='Trade data file')

    args = parser.parse_args()
    TRAIN_START_DATE = args.start_date
    TRAIN_END_DATE = args.end_date

    processed_full = pd.read_csv(args.data_file)
    processed_full['date'] = pd.to_datetime(processed_full['date']).dt.strftime('%Y-%m-%d') 
    trade = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)

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

    check_and_make_directories([TRAINED_MODEL_DIR])

    # Environment
    e_trade_gym_switch = StockTradingEnv(df=trade, **env_kwargs)

    # selected agents
    ppoReal1 = MyPPO.load('../KL_version/trained_models/' + 'ppoReal1')
    a2cEma1 = MyA2C.load('../KL_version/trained_models/' + 'a2cEma1')
    a2cMean4 = MyA2C.load('../KL_version/trained_models/' + 'a2cMean4')
    ppoMin3 = MyPPO.load('../KL_version/trained_models/' + 'ppoMin3')
    a2cEma4 = MyA2C.load('../KL_version/trained_models/' + 'a2cEma4')
    ppoReal4 = MyPPO.load('../KL_version/trained_models/' + 'ppoReal4')
    a2cMax2 = MyA2C.load('../KL_version/trained_models/' + 'a2cMax2')
    ppoMax3 = MyPPO.load('../KL_version/trained_models/' + 'ppoMax3')
    ppoMax1 = MyPPO.load('../KL_version/trained_models/' + 'ppoMax1')

    # Backtesting
    ppo_switch = PPO_Switch(switchWindows=[1], hmax=env_kwargs['hmax'], stocksDimension=stock_dimension)

    epochs = 10
    lr = 0.001
    # moe model
    model = [ppoReal1, a2cEma1, a2cMean4, ppoMin3, a2cEma4, ppoReal4, a2cMax2, ppoMax3, ppoMax1]

    moe_model = MoE(model)
    optimizer_moe = optim.Adam(moe_model.parameters(), lr=lr)
    # print(moe_model.parameters())
    # for param in moe_model.parameters():
    #     print(type(param), param.size())

    for epoch in range(epochs):
        print('------------epoch:', epoch)
        # train
        df_result_ppo, df_actions_ppo, moe_model, optimizer_moe = ppo_switch.DRL_prediction(moe_model=moe_model, optimizer_moe=optimizer_moe, environment=[e_trade_gym_switch])
        # print("==============Get Backtest Results===========")
        perf_stats_all = backtest_stats(account_value=df_result_ppo)

    # save moe_model
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    torch.save({
        'model_state_dict': moe_model.state_dict(),
        'optimizer_state_dict': optimizer_moe.state_dict(),
    }, './trained_models/trained_moe_model_' + str(current_time) + '.pth')
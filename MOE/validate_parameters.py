import os
import pandas as pd
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
from MOE.ensemble_final import Ensemble_Final

# validate
DIR = '../Data/data/'

TRADE_START_DATE = '2014-01-05'
TRADE_END_DATE = '2015-01-06'
DATA_NAME = 'DOW29/full-DOW29_Real'

# TRADE_START_DATE = '2015-01-06'
# TRADE_END_DATE = '2016-01-07'
# DATA_NAME = 'HS29/full-HS29_Real'

# TRADE_START_DATE = '2021-11-11'
# TRADE_END_DATE = '2022-11-11'
# DATA_NAME = 'CRYPTO29/full-CRYPTO29_Real'
#
# TRADE_START_DATE = '2019-01-03'
# TRADE_END_DATE = '2020-01-04'
# DATA_NAME = 'HK29/full-HK29_Real'
#
# TRADE_START_DATE = '2017-01-02'
# TRADE_END_DATE = '2018-01-03'
# DATA_NAME = 'NYSE29/full-NYSE29_Real'
#
# TRADE_START_DATE = '2021-01-04'
# TRADE_END_DATE = '2022-01-05'
# DATA_NAME = 'FTSE29/full-FTSE29_Real'


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

    check_and_make_directories([TRAINED_MODEL_DIR])

    # Environment
    e_trade_gym_switch = StockTradingEnv(df=trade, **env_kwargs)

    # agents
    ppoReal1 = MyPPO.load('../KL_version/trained_models/0.0005_256/' + 'ppoReal1')
    a2cEma1 = MyA2C.load('../KL_version/trained_models/0.0005_256/' + 'a2cEma1')
    a2cMean4 = MyA2C.load('../KL_version/trained_models/0.0005_256/' + 'a2cMean4')
    ppoMin3 = MyPPO.load('../KL_version/trained_models/0.0005_256/' + 'ppoMin3')
    a2cEma4 = MyA2C.load('../KL_version/trained_models/0.0005_256/' + 'a2cEma4')
    ppoReal4 = MyPPO.load('../KL_version/trained_models/0.0005_256/' + 'ppoReal4')
    a2cMax2 = MyA2C.load('../KL_version/trained_models/0.0005_256/' + 'a2cMax2')
    ppoMax3 = MyPPO.load('../KL_version/trained_models/0.0005_256/' + 'ppoMax3')
    ppoMax1 = MyPPO.load('../KL_version/trained_models/0.0005_256/' + 'ppoMax1')

    # Backtesting
    # load moe 
    model_name = 'trained_moe_model.pth'
    moe_model = MoE(
        trained_experts=[ppoReal1, a2cEma1, a2cMean4, ppoMin3, a2cEma4, ppoReal4, a2cMax2, ppoMax3, ppoMax1])
    checkpoint = torch.load('./trained_models/' + model_name)
    moe_model.load_state_dict(checkpoint['model_state_dict'])
    moe_model.eval()

    for a in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        ppo_switch = Ensemble_Final(alpha=a, switchWindows=[6,9,10], hmax=env_kwargs['hmax'],
                                      stocksDimension=stock_dimension)

        df_result_ppo, df_actions_ppo = ppo_switch.DRL_prediction(environment=[e_trade_gym_switch], model_moe=moe_model)


        # print("==============Get Backtest Results===========")
        # perf_stats_all = backtest_stats(account_value=df_result_ppo)

       
        # print(df_result_ppo.iloc[0])
        print('------>', a, df_result_ppo.iloc[-1])

        # save_dir = "./alphaResults/" + str(DATA_NAME[:-9]) + '/'
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        #
        # df_result_ppo.to_csv(save_dir + str(a) + "_alpha_cw.csv", index=False)

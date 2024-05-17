import os

import pandas as pd
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import TensorboardCallback
from stable_baselines3.common.logger import configure
from finrl.main import check_and_make_directories
from finrl.config import TRAINED_MODEL_DIR, RESULTS_DIR
from my_a2c import MyA2C
from my_ppo import MyPPO

# Contestants are welcome to split the data in their own way for model tuning
TRAIN_START_DATE = 'TRAIN_START_DATE'
TRAIN_END_DATE = 'TRAIN_END_DATE'
processed_full = pd.read_csv('./Data/full-mixed-Real.csv', low_memory=False) 
processed_full['date'] = pd.to_datetime(processed_full['date']).dt.strftime('%Y-%m-%d')  
train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)

# Environment configs
stock_dimension = len(train.tic.unique())

#######################################
INDICATORS1 = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma"]
state_space = 1 + 2 * stock_dimension + len(INDICATORS1) * stock_dimension  
################################

print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS1,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

# PPO configs
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.0005,
    "batch_size": 256,  
}

# A2C configs
A2C_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.0005, 
}


def train_model(
        model, tb_log_name, total_timesteps=5000, model_name="", models=None, agent_name=""
):  # this function is static method, so it can be called without creating an instance of the class
    model = model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=tb_log_name,
        callback=TensorboardCallback(),
        model_name=model_name,  
        models=models,  
        agent_name=agent_name  
    )
    return model

if __name__ == '__main__':
    check_and_make_directories([TRAINED_MODEL_DIR])

    # types = ['ppo_real', 'ppo_max', 'ppo_min', 'ppo_mean', 'ppo_ema']
    types = ['a2c_real', 'a2c_max', 'a2c_min', 'a2c_mean', 'a2c_ema']
    cur_type = types[0]  

    # agents = ['ppoReal1', 'ppoReal2', 'ppoReal3', 'ppoReal4']
    # agents = ['ppoMax1', 'ppoMax2', 'ppoMax3', 'ppoMax4']
    # agents = ['ppoMin1', 'ppoMin2', 'ppoMin3', 'ppoMin4']
    # agents = ['ppoMean1', 'ppoMean2', 'ppoMean3', 'ppoMean4']
    # agents = ['ppoEma1', 'ppoEma2', 'ppoEma3', 'ppoEma4']

    agents = ['a2cReal1', 'a2cReal2', 'a2cReal3', 'a2cReal4']
    # agents = ['a2cMax1', 'a2cMax2', 'a2cMax3', 'a2cMax4']
    # agents = ['a2cMin1', 'a2cMin2', 'a2cMin3', 'a2cMin4']
    # agents = ['a2cMean1', 'a2cMean2', 'a2cMean3', 'a2cMean4']
    # agents = ['a2cEma1', 'a2cEma2', 'a2cEma3', 'a2cEma4']

    # for agent in agents:
    #     cur_agent_name = agent
    cur_agent_name = agents[0]  
    kl_coef = 25

    models = []
    if cur_agent_name[-1] == '1':
        pass
    else:
        index = 1
        while index < int(cur_agent_name[-1]):
            # models.append(MyPPO.load(TRAINED_MODEL_DIR + '/' + agents[index-1]))   # ppo load
            models.append(MyA2C.load(TRAINED_MODEL_DIR + '/' + agents[index-1]))   # a2c load
            index += 1

    # Environment
    e_train_gym = StockTradingEnv(df=train, **env_kwargs, model_name=cur_agent_name, mode='train')
    env_train, _ = e_train_gym.get_sb_env()

    # my_model = MyPPO(                              ##  ppo
    #     policy="MlpPolicy",
    #     env=env_train,
    #     model_name=cur_agent_name,
    #     **PPO_PARAMS,
    #     models=models,  
    #     kl_coef=kl_coef 
    # )
    my_model = MyA2C(                            ## a2c
        policy="MlpPolicy",
        env=env_train,
        model_name=cur_agent_name,
        **A2C_PARAMS,
        models=models,  
        kl_coef=kl_coef 
    )

    # set up logger
    tmp_path = RESULTS_DIR + '/' + cur_agent_name
    new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    my_model.set_logger(new_logger_ppo)

    trained_ppo = train_model(model=my_model,
                              tb_log_name=cur_agent_name,
                              total_timesteps=80000,
                              model_name=cur_agent_name,
                              models=models,
                              agent_name=cur_agent_name)

    # save
    # trained_ppo.save(TRAINED_MODEL_DIR + '/' + cur_agent_name)

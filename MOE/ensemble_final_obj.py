import statistics
import numpy as np
import torch
from torch import optim, nn

from MOE.MoE import MoE


class Ensemble_Final_Obj:
    def __init__(self, environment, model_moe, stocksDimension, alpha, switchWindows, hmax=100):
        self.switchWindows = switchWindows
        self.stocksDim = stocksDimension
        self.hamx =hmax
        self.last_obs = None
        self.alpha = alpha
        self.environment = environment
        self.model_moe = model_moe
        pass

    def init_predict(self):
        self.ppo_switch_env, self.ppo_switch_obs = self.environment[0].get_sb_env()  
        self.account_memory = None  # This help avoid unnecessary list creation
        self.actions_memory = None  # optimize memory consumption
        self.ppo_switch_env.reset()
        self.max_steps = len(self.environment[0].df.index.unique()) - 1
        self.switchWindow = self.switchWindows[0]

    def DRL_prediction(self, i):
        """
            make a prediction and get results
            :param model: (list<object>) multiple different model
            :param environment: (list<object>) a final test environment and multiple models corresponding environment
            :param deterministic: (bool) Whether or not to return deterministic actions.
            :return: (df) cumulative wealth and actions record in different periods
        """
        with torch.no_grad():
            finalAction = self.model_moe(self.ppo_switch_obs)
        finalAction = (finalAction.detach().numpy(), None)
        # print(finalAction)

        # Sparse the chooseAction (invest in only one stock)
        finalAction = self.sparse_action(self.environment[0], finalAction, self.stocksDim)  
        # print(finalAction)

        if i >= max(self.switchWindows):
            if i % self.switchWindow == 0:
                cwList = [0 for _ in range(len(self.switchWindows))]
                for j in range(len(self.switchWindows)):
                    pre_cw = (self.environment[0].asset_memory[i] - self.environment[0].asset_memory[i - self.switchWindows[j]]) / self.switchWindows[j]
                    std = self.get_modelRewardStd(self.environment[0], self.switchWindows[j])
                    cwList[j] += self.alpha * pre_cw + (1-self.alpha) * std
                self.switchWindow = self.switchWindows[np.argmax(cwList)]
            else:
                finalAction = [np.array([0 for _ in range(self.stocksDim)])]

        return finalAction, self.ppo_switch_obs[0][0]


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

    def get_modelNowCW(self, environments, day):
        """
        Retrieves the current cumulative wealth from the given environments for a specific day.

        :param environments: (list<object>) List of environment objects
        :param day: (int) Day index
        :return: (list<int>) List of cumulative wealth for the given day
        """
        res = [0 for _ in range(len(environments) - 1)]
        # Iterate over the environments (except the first one)
        for i in range(len(environments) - 1):
            res[i] = environments[i + 1].asset_memory[day]
        return res

    def get_modelReward(self, environments, day, interval):
        """
        Calculates the reward based on the change in cumulative wealth between the current day and a specified interval.

        :param environments: (list) List of environment objects
        :param day: Current day index
        :param interval: Time interval
        :return: List of rewards for the given interval
        """
        res = [0 for _ in range(len(environments) - 1)]
        for i in range(len(environments) - 1):
            res[i] = environments[i + 1].asset_memory[day] - environments[i + 1].asset_memory[day - interval]
        return res

    def get_modelRewardStd(self, environment, interval):
        rewards = environment.asset_memory[-interval-1:]
        for i in range(len(rewards)-1):
            rewards[i] = rewards[i+1] - rewards[i]
        res = statistics.stdev(rewards[:-1])
        return res
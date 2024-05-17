import statistics
import numpy as np
import torch
from torch import optim, nn
from MOE.MoE import MoE


class Ensemble_Final:
    def __init__(self, stocksDimension, alpha, switchWindows, hmax=100):
        self.switchWindows = switchWindows
        self.stocksDim = stocksDimension
        self.hamx =hmax
        self.last_obs = None
        self.alpha = alpha
        pass

    def DRL_prediction(self, environment, model_moe=None):
        """
            make a prediction and get results
            :param model: (list<object>) multiple different model
            :param environment: (list<object>) a final test environment and multiple models corresponding environment
            :param deterministic: (bool) Whether or not to return deterministic actions.
            :return: (df) cumulative wealth and actions record in different periods
        """
        ppo_switch_env, ppo_switch_obs = environment[0].get_sb_env() 

        account_memory = None  # This help avoid unnecessary list creation
        actions_memory = None  # optimize memory consumption

        ppo_switch_env.reset()

        max_steps = len(environment[0].df.index.unique()) - 1

        switchWindow = self.switchWindows[0]

        for i in range(len(environment[0].df.index.unique())):

            # moe 
            with torch.no_grad():
                finalAction = model_moe(ppo_switch_obs)
            finalAction = (finalAction.detach().numpy(), None)
            # print(finalAction)

            # Sparse the chooseAction (invest in only one stock)
            finalAction = self.sparse_action(environment[0], finalAction, self.stocksDim)
            # print(finalAction)

            if i >= max(self.switchWindows):
                if i % switchWindow == 0:
                    cwList = [0 for _ in range(len(self.switchWindows))]
                    for j in range(len(self.switchWindows)):
                        pre_cw = (environment[0].asset_memory[i] - environment[0].asset_memory[i - self.switchWindows[j]]) / self.switchWindows[j]
                        std = self.get_modelRewardStd(environment[0], self.switchWindows[j])
                        cwList[j] += self.alpha * pre_cw + (1-self.alpha) * std 
                    switchWindow = self.switchWindows[np.argmax(cwList)]
                else:
                    finalAction = [np.array([0 for _ in range(self.stocksDim)])]

            ppo_switch_obs, _, dones, _ = ppo_switch_env.step(finalAction)

            if i == max_steps - 1:  # more descriptive condition for early termination to clarify the logic
                account_memory = ppo_switch_env.env_method(method_name="save_asset_memory")
                actions_memory = ppo_switch_env.env_method(method_name="save_action_memory")
            # add current state to state memory
            # state_memory=test_env.env_method(method_name="save_state_memory")
            if dones[0]:
                print("hit end!")
                break
        return account_memory[0], actions_memory[0]

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
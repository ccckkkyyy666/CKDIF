import datetime
import statistics
import numpy as np
import torch
from torch import optim, nn
from MOE.MoE import MoE

class PPO_Switch:
    def __init__(self, stocksDimension, switchWindows, hmax=100):
        self.switchWindows = switchWindows
        self.stocksDim = stocksDimension
        self.hamx = hmax
        pass

    def DRL_prediction(self, moe_model, optimizer_moe, environment):
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

        for i in range(len(environment[0].df.index.unique())):
            optimizer_moe.zero_grad()
            outputs_moe = moe_model(ppo_switch_obs)
            # print('--moe output:', outputs_moe)

            pre_cash = ppo_switch_obs[0][0]
            pre_price = ppo_switch_obs[0][1:30]
            pre_hold = ppo_switch_obs[0][30:59]

            finalAction = (outputs_moe.detach().numpy(), None)
            ppo_switch_obs, _, dones, _ = ppo_switch_env.step(finalAction)

            now_cash = ppo_switch_obs[0][0]
            now_price = ppo_switch_obs[0][1:30]
            now_hold = ppo_switch_obs[0][30:59]

            reward = now_cash + np.dot(now_hold, now_price) - (pre_cash + np.dot(pre_hold, pre_price))

            loss_moe = torch.tensor(-reward, dtype=torch.float64, requires_grad=True)

            print('loss-moe:', loss_moe)
            loss_moe.backward()  
            optimizer_moe.step()

            if i == max_steps - 1:  # more descriptive condition for early termination to clarify the logic
                account_memory = ppo_switch_env.env_method(method_name="save_asset_memory")
                actions_memory = ppo_switch_env.env_method(method_name="save_action_memory")

            # add current state to state memory
            # state_memory=test_env.env_method(method_name="save_state_memory")
            if dones[0]:
                print("hit end!")
                break
        return account_memory[0], actions_memory[0], moe_model, optimizer_moe

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
        newAction = np.array(heldQuantity) / -self.hamx   # sell holding

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

    def get_modelRewardStd(self, environments, interval):
        res = [0 for _ in range(len(environments) - 1)]
        for i in range(len(environments) - 1):
            agent_rewards = environments[i + 1].asset_memory[-interval:]
            res[i] = statistics.stdev(agent_rewards)
        return res

import pandas as pd
import numpy as np
from typing import Union
import time

"""
strategy introduction:
    - strategy_1: buy or sell the amount of shares of the actor output
    - strategy_2: buy, sell or hold fixed of the actor output
    - strategy_3: TODO
"""
class StrategySelector():

    """convert actor's output to real action"""
    def __init__(self, **strategy_kwargs):

        self.strategy_name      : str   = strategy_kwargs['strategy_name']
        self.action_dim         : int
        self.initial_balance    : float = strategy_kwargs['initial_balance']
        self.max_qty_per_order  : float = 10000
        self.pyramiding         : int   = 3 # TODO: implement pyramiding
        self.max_position_size  : float = None

        self.actor_output       : Union[int, float, list]
        self.current_price      : float
        self.position_size      : float
        self.real_action        : float
        self.direction          : str

        self.reward_function    : str   = strategy_kwargs['reward_function']
        self.reward             : float
        self.cumulative_reward  : float = 0
        self.initialize_strategy()

    def initialize_strategy(self) -> None:
        """
        initialize the strategy's required parameters
        """
        strategy_to_dim = {'strategy_1': 1, 
                           'strategy_2': 3, 
                           'strategy_3': 1}
        strategy_to_actor_activation = {'strategy_1': 'tanh', 
                                        'strategy_2': 'softmax', 
                                        'strategy_3': 'tanh'}

        self.action_dim       = strategy_to_dim.get(self.strategy_name, None)
        self.actor_activation = strategy_to_actor_activation.get(self.strategy_name, None)
        if self.action_dim is None or self.actor_activation is None:
            raise ValueError(f"Invalid strategy name: {self.strategy_name}")
        
        print('########## Strategy Selector ##########')
        print(f'initialize strategy:        {self.strategy_name}')
        print(f'reward_function:            {self.reward_function}')
        print(f'action_dim:                 {self.action_dim}')
        print(f'actor_activation:           {self.actor_activation}')
        return

    def convert_to_real_action(self, **kwargs) -> tuple:
        """
        convert actor's ouput to real action
        :param kwargs: 
            - current_price         (float):            current price
            - actor_output          (int, float, list): actor's output
            - position_size   (float):            current holding quantity
        """
        self.current_price       = kwargs['current_price']
        self.actor_output        = kwargs['actor_output']
        self.position_size       = kwargs['position_size']
        self.max_position_size   = (self.max_qty_per_order * self.pyramiding) / self.current_price

        # stratetgy_1: buy or sell the amount of shares of the actor output
        if self.strategy_name == 'strategy_1':
            self.actor_output    = float(self.actor_output[0][0][0])

            target_position_size = (self.max_qty_per_order * self.actor_output) / self.current_price
            target_position_size = np.clip(target_position_size, -self.max_position_size, self.max_position_size)
            self.real_action     = target_position_size - self.position_size # base currency

        # stratetgy_2: buy, sell or hold max_qty_per_order of highest prob of the actor output(no pyramiding)
        if self.strategy_name == 'strategy_2':
            action_probs = self.actor_output[0][0].numpy() / np.sum(self.actor_output[0][0].numpy()) # make sure the sum of probs is 1
            self.actor_output = np.argmax(action_probs) - 1
            # print(f'position size: {self.position_size}')
            # print(f'max_position_size: {self.max_position_size}')
            if (self.actor_output == -1) and (self.position_size >= -self.max_position_size):
                self.real_action = (self.max_qty_per_order * self.actor_output)  / self.current_price # base currency
            elif self.actor_output == 1 and self.position_size <= self.max_position_size:
                self.real_action = (self.max_qty_per_order * self.actor_output)  / self.current_price # base currency
            elif self.actor_output == 0:
                self.real_action = 0
            else:
                self.real_action = 0

        # strategy_3: TODO
        if self.strategy_name == 'strategy_3':
            pass

        # print(f'real action : {self.real_action}')
        # time.sleep(0.2)
        self.direction = 'long' if self.real_action > 0 else 'short' if self.real_action < 0 else 'hold'
        return (self.direction, self.real_action)

    def calc_reward(self, **kwargs) -> tuple:
        """
        calculate the reward
        :param kwargs:
            - next_price:       (float):        next price
            - past_sample_df:   (pd.DataFrame): past sample data, used to calculate Sharpe ratio
                - columns: ['datetime', 'close', 'volume']
        """
        past_sample_df  = kwargs['past_sample_df'].copy()
        self.next_price = kwargs['next_price']

        # reward function_1 (quote currency return of the action)
        if self.reward_function == 'reward_function_1':
            self.reward = self.real_action * (self.next_price - self.current_price)
                
        # reward function_2 (quote currency return of the whole position)
        if self.reward_function == 'reward_function_2':
            self.reward = self.position_size * (self.next_price - self.current_price)
        
        # reward_function_3 (return(%) - lambda * drawdown(%))
        if self.reward_function == 'reward_function_3':
            ret      = past_sample_df.iloc[-1]['return']
            drawdown = past_sample_df.iloc[-1]['drawdown_pct']
            LAMBDA = 0.1
            self.reward = ret + LAMBDA * drawdown

        # reward function_4 (reward = sharpe ratio)
        if self.reward_function == 'reward_function_4':
            if len(past_sample_df) < 64:
                self.reward = 0 # self.real_action * (self.next_price - self.current_price)
            elif len(past_sample_df) >= 64:
                past_sample_df['return'] = past_sample_df['net_worth'].pct_change()
                # time.sleep(0.1)
                past_sample_df.dropna(inplace = True)
                past_sample_df = past_sample_df.iloc[-64:] # 用最近的64個數據計算sharpe ratio
                mean = past_sample_df['return'].mean()
                std  = past_sample_df['return'].std()
                sharpe_ratio = mean / std
                print(sharpe_ratio)
                self.reward  = sharpe_ratio

        self.cumulative_reward += self.reward
        return self.reward, self.cumulative_reward

if __name__ == '__main__':
    pass














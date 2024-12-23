import time
import gym
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.python as tfp
from tensorflow.python.keras import layers
from tensorflow.python.keras.optimizers import adam_v2
from strategy import StrategySelector
from typing import Union


debug = False

class TradingEnv(gym.Env):

    def __init__(self, data: pd.DataFrame, strategy: StrategySelector):
        super(TradingEnv, self).__init__()
        self.data               : pd.DataFrame     = data
        self.strategy           : StrategySelector = strategy
        self.current_step       : int              = 0
        self.reward             : float            = 0
        self.cumulative_reward  : float            = 0
        self.position_size      : float            = 0
        self.net_worth          : int              = self.strategy.initial_balance
        self.max_step           : int              = len(data) - 1

    def reset(self):    
        self.current_step        = 0
        self.reward              = 0
        self.cumulative_reward   = 0
        self.position_size       = 0
        self.net_worth           = self.strategy.initial_balance
        
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:

        """return state: [o, h, l, c, v]"""
        """TODO: add another observation"""
        numeric_data = self.data.select_dtypes(include = [np.number]) # elminate Date column
        obs = (numeric_data.iloc[self.current_step].values - numeric_data.mean()) / numeric_data.std() # normalize data
        obs_col = ['Open', 'High', 'Low', 'Close', 'Volume']
        obs = obs[obs_col]
        output_state = obs
        return output_state
    
    def step(self, action: Union[int, float, list, np.ndarray]) -> tuple:
        

        current_price = self.data.iloc[self.current_step]['Close']
        next_price    = self.data.iloc[self.current_step + 1]['Close']
        action_kwargs = {
            'current_price'       : self.data.iloc[self.current_step]['Close'],
            'actor_output'        : action,
            'position_size'       : self.position_size,
        }
        direction, real_action = self.strategy.convert_to_real_action(**action_kwargs)
        print(f'current step:   {self.current_step}')                                   if debug == True else None
        print(f'taking action:  {direction} {real_action}')                             if debug == True else None

        self.current_step       += 1
        self.position_size      += real_action
        self.net_worth          += self.position_size * (next_price - current_price)
        self.data.loc[self.current_step, 'net_worth']     = self.net_worth
        self.data.loc[self.current_step, 'return']        = (self.net_worth - self.data.loc[self.current_step - 1, 'net_worth']) / self.data.loc[self.current_step - 1, 'net_worth']
        self.data.loc[self.current_step, 'position_size'] = self.position_size
        self.data.loc[self.current_step, 'drawdown']      = self.net_worth - self.data['net_worth'].max()
        self.data.loc[self.current_step, 'drawdown_pct']  = (self.net_worth - self.data['net_worth'].max()) / self.data['net_worth'].max()

        self.past_sample_df = self.data.iloc[: self.current_step + 1][['datetime', 'Close', 'net_worth', 'return', 'position_size', 'drawdown', 'drawdown_pct']]
        # print(f'past sample df: {self.past_sample_df}')                                 if debug == True else None
        # time.sleep()                                                                  if debug == True else None

        reward_kwargs = {
            'next_price'          : next_price,
            'past_sample_df'      : self.data.iloc[: self.current_step + 1],
        }
        self.reward, self.cumulative_reward = self.strategy.calc_reward(**reward_kwargs)

        print(f'next price:     {next_price}')                                          if debug == True else None
        print(f'reward:         {self.reward}')                                         if debug == True else None
        print(f'net_worth:      {self.net_worth}')                                      if debug == True else None

        done = self.current_step >= self.max_step
        return self._get_observation(), self.reward, done, {}

    def render(self):
        print(f'step: {self.current_step}, Net_Worth: {self.net_worth}')

class Actor(tfp.keras.Model):

    def __init__(self, action_dim: int, activation: str):
        super(Actor, self).__init__()
        self.fc1 = layers.Dense(256, activation = 'relu')
        self.fc2 = layers.Dense(256, activation = 'relu')
        self.fc3 = layers.Dense(action_dim, activation)
        print(f'actor initialize: action_dim = {action_dim}, activation = {activation}')if debug == True else None

    def call(self, state: np.ndarray) -> tf.Tensor:
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        print(f'actor raw ouptut: {x}')                                                 if debug == True else None
        print(f'actor raw ouptut shape: {x.shape}')                                     if debug == True else None
        return x # output action

class Critic(tfp.keras.Model):

    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = layers.Dense(256, activation = 'relu')
        self.fc2 = layers.Dense(256, activation = 'relu')
        self.fc3 = layers.Dense(1)

    def call(self, state: np.ndarray, action: Union[int, np.ndarray]) -> tf.Tensor:
        x = layers.Concatenate()([state, action])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x # output value estimation: Q(s, a)

class DDPGAgent():
    
    def __init__(self, strategy: StrategySelector):
        
        self.strategy                = strategy
        self.actor                   = Actor(self.strategy.action_dim, activation = self.strategy.actor_activation)
        self.target_actor            = Actor(self.strategy.action_dim, activation = self.strategy.actor_activation)
        self.critic                  = Critic()
        self.target_critic           = Critic()
        self.actor.optimizer         = adam_v2.Adam(learning_rate = 1e-4)
        self.critic.optimizer        = adam_v2.Adam(learning_rate = 1e-4)
        self.target_actor.optimizer  = adam_v2.Adam(learning_rate = 1e-4)
        self.target_critic.optimizer = adam_v2.Adam(learning_rate = 1e-4)
        self.buffer                  = []
        self.batch_size              = 64
        self.gamma                   = 0.99

    def update_target_networks(self, tau = 0.005):
        for target_param, param in zip(self.target_actor.trainable_variables, self.actor.trainable_variables):
            target_param.assign(tau * param + (1 - tau) * target_param)
        for target_param, param in zip(self.target_critic.trainable_variables, self.critic.trainable_variables):
            target_param.assign(tau * param + (1 - tau) * target_param)

    def select_action(self, state) -> np.ndarray:

        action = self.actor(np.array([state]))
        print('selecting action...')                                                    if debug == True else None
        print(f'current state: {[i for i in state]}')                                   if debug == True else None
        return action

    def store_transition(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) > 10000:
            self.buffer.pop(0)

    def train(self):
        if len(self.buffer) < self.batch_size:
            return # start training after buffer size > batch size

        indices = np.random.choice(len(self.buffer), self.batch_size)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = np.array(states)
        print(f'actions shape: {np.array(actions)}')                                    if debug == True else None
        print(f'actions shape: {np.array(actions).shape}')                              if debug == True else None
        actions     = tf.reshape(np.array(actions), (64, self.strategy.action_dim))
        rewards     = np.array([float(r) if isinstance(r, np.ndarray) else r for r in rewards]).reshape(-1, 1)
        next_states = np.array(next_states)
        dones       = np.array(dones).reshape(-1, 1)

        # compute target Q values
        with tf.GradientTape() as tape:
            print(f'next_states shape: {next_states.shape}')                            if debug == True else None
            target_actions  = self.target_actor(next_states)
            print(f'target_actions shape: {target_actions.shape}')                      if debug == True else None
            target_q_values = self.target_critic(next_states, target_actions)
            q_targets       = rewards + self.gamma * target_q_values * (1 - dones)
            print(f'states shape: {states.shape}')                                      if debug == True else None
            print(f'actions shape: {actions.shape}')                                    if debug == True else None
            q_values        = self.critic(states, actions)
            critic_loss     = tf.reduce_mean(tf.square(q_targets - q_values))

        # 計算梯度(d(critic_loss)/d(critic.trainable_variables))
        critic_gradients    = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        # compute actor loss
        with tf.GradientTape() as tape:
            tape.watch(self.actor.trainable_variables)
            actions_pred = self.actor(states)
            q_values = self.critic(states, actions_pred)
            actor_loss = -tf.reduce_mean(q_values)

        # 計算梯度(d(actor_loss)/d(actor.trainable_variables))
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        # update target networks
        self.update_target_networks()


def train_ddpg():

    data = pd.read_csv(f'C:/Users/jack/Desktop/TMBA/train_data.csv')
    strategy_list = ['strategy_1', 'strategy_2']
    reward_list   = ['reward_function_1', 'reward_function_2', 'reward_function_4']
    for strategy_name in strategy_list:
        for reward_function in reward_list:
            strategy_kwargs = {
                'strategy_name'     : strategy_name,
                'reward_function'   : reward_function,
                'initial_balance'   : 10000, 
                'max_qty_per_order' : 10000,
                'pyramiding'        : 3}
            strategy    = StrategySelector(**strategy_kwargs)
            env         = TradingEnv(data = data, strategy = strategy)
            agent       = DDPGAgent(strategy = strategy)

            for episode in range(1, 50):
                print(f'--------------------- Start episode: {episode}---------------------')
                state = env.reset()
                strategy.cumulative_reward = 0
                done  = False
                while not done:
                    initial_noise_scale = 0.1
                    decay_rate          = 0.99
                    noise = tf.random.normal(shape = [1, strategy.action_dim], mean = 0.0, stddev = 0.1)
                    noise_scale = max(0.1, initial_noise_scale * (decay_rate ** episode))
                    action = agent.select_action(state) + noise_scale * noise
                    next_state, reward, done, _ = env.step([action])
                    agent.store_transition([state, action, reward, next_state, done])
                    agent.train()
                    state = next_state

                    print(f'noise:  {noise_scale * noise}')                                 if debug == True else None
                    print(f'position size: {env.position_size}')                            if debug == True else None
                    print(f'action: {action}')                                              if debug == True else None
                    time.sleep(0.1)                                                         if debug == True else None

                    print(f'\r step: {env.current_step} --> cumulative reward: {env.cumulative_reward}', end = ' ')
                env.past_sample_df.to_csv(f'C:/Users/jack/Desktop/TMBA/{strategy.strategy_name}_{strategy.reward_function}_train_result.csv', index = False)
                print(f'##### End of episode: {episode} /// Net Worth {env.net_worth} #####')
                agent.actor.save_weights(f'actor_{strategy_name}_{reward_function}', save_format = 'tf')
                agent.critic.save_weights(f'critic_{strategy_name}_{reward_function}', save_format = 'tf')    
                # time.sleep(5)

if __name__ == '__main__':
    
    train_ddpg()
from IPython.display import clear_output
import numpy as np
from frozen_lake_env import FrozenLakeEnv
from numpy.random import random, choice
import matplotlib.pyplot as plt
import time


class SARSA:
  def __init__(self,env = FrozenLakeEnv(map_name="4x4")):
    self.env = env
    self.no_states = self.env.observation_space.n
    self.no_actions = self.env.action_space.n
    self.Q = np.zeros((self.no_states, self.no_actions))
    self.returns = []


  def epsilon_greedy(self, state,epsilon):
    new_action = self.env.action_space.sample()

    greedy_actions = np.argmax(self.Q[state,:])

    explore = (random() < epsilon)

    if explore:
      return new_action # choose a random action
    else:
      return  greedy_actions #choose max value action

  def train(self,episodes = 1000,gamma = 1.0,alpha = 0.1,epsilon = 1.0,decay_rate = 0.01):

      for episode in range(episodes):

        state = self.env.reset()
        epsilon = 0.001 + 0.999 * np.exp(-decay_rate * episode)



        action = self.epsilon_greedy(state = state,epsilon = epsilon)


        rewards = [None]  # No first return

        #Prevent entry into local optimal solution
        max_steps = 500
        for step in range(max_steps):

          new_state, reward, done, info = self.env.step(action)
          new_action = self.epsilon_greedy(state = new_state, epsilon=epsilon)

          rewards.append(reward)
          #Update Q table
          self.Q[state,action] = self.Q[state,action] + alpha * (reward + gamma * self.Q[new_state, new_action] - self.Q[state, action])
          state, action = new_state, new_action
          if done:
            break
       #PLOT the success_rate
        T = len(rewards)
        G = 0
        # t = T-2, T-3, ..., 0
        t = T - 2
        while t >= 0:
          G = rewards[t + 1] + gamma * G
          t = t - 1
        self.returns.append(G)
      window_size = 100
      averaged_returns = np.zeros(len(self.returns) - window_size + 1)

      for i in range(len(averaged_returns)):
        averaged_returns[i] = np.mean(self.returns[i:i + window_size])

      plt.plot(averaged_returns, linewidth=2)
      plt.xlabel("Episode")
      plt.ylabel("average of first returns (window_size={})".format(window_size))
      plt.show()
      print(self.Q)


  def test(self,episodes_test=1):
      time.sleep(2)

      for episode in range(episodes_test):
        state = self.env.reset()
        done = False
        print('failed ', episode + 1)

        time.sleep(1.5)

        steps = 0
        while not done:
          clear_output(wait=True)
          self.env.render()
          time.sleep(0.3)

          action = np.argmax(self.Q[state])
          state, reward, done, _ = self.env.step(action)
          steps += 1

        clear_output(wait=True)
        self.env.render()

        if reward == 1:
          print(f'steps are {steps} .')
          time.sleep(2)
        else:
          print('Sorry, try again!')
          time.sleep(2)
        clear_output(wait=True)

if __name__ == '__main__':
  agent = SARSA(env = FrozenLakeEnv(map_name="10x10"))
  agent.train(episodes = 1000,gamma = 1.0,alpha = 0.1,epsilon = 1.0,decay_rate = 0.01)
  # agent.test(episodes_test=1)


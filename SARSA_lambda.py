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
    values = self.Q[state, :]
    max_value = max(values)
    no_actions = len(values)

    greedy_actions = [a for a in range(no_actions) if values[a] == max_value]

    explore = (random() < epsilon)

    if explore:
      return choice([a for a in range(no_actions)])
    else:
      return choice([a for a in greedy_actions])

  def train(self,episodes = 1000,gamma = 1.0,alpha = 0.1,epsilon = 1.0,decay_rate = 0.1, eligibility_decay = 0.3):

      for episode in range(episodes):

        state = self.env.reset()
        epsilon = 0.001 + 0.999 * np.exp(-decay_rate * episode)


        action = self.epsilon_greedy(state = state,epsilon = epsilon)

        R = [None]  # No first return
        E = np.zeros((self.no_states, self.no_actions))

        while True:

          E = eligibility_decay*gamma * E
          E[state, action] += 1

          new_state, reward, done, info = self.env.step(action)
          new_action = self.epsilon_greedy(state =new_state, epsilon=epsilon)

          R.append(reward)

          delta = reward + gamma * self.Q[new_state, new_action] - self.Q[state, action]
          self.Q = self.Q + alpha * delta * E

          state, action = new_state, new_action
          if done:
            break
       #PLOT the success_rate
        T = len(R)
        G = 0
        # t = T-2, T-3, ..., 0
        t = T - 2
        while t >= 0:
          G = R[t + 1] + gamma * G
          t = t - 1
        self.returns.append(G)
      window_size = 100
      averaged_returns = np.zeros(len(self.returns) - window_size + 1)

      for i in range(len(averaged_returns)):
        averaged_returns[i] = np.mean(self.returns[i:i + window_size])

      plt.plot(averaged_returns, linewidth=2)
      plt.xlabel("Episode")
      plt.ylabel("Moving average of first returns (window_size={})".format(window_size))
      plt.show()
      print(self.Q)


  def test(self,episodes_test=1):
      global reward
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
  agent.train(episodes = 1000,gamma = 1.0,alpha = 0.1,epsilon = 1.0,decay_rate = 0.1, eligibility_decay = 0.3)
  # agent.test(episodes_test=1)



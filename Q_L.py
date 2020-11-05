import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
from frozen_lake_env import FrozenLakeEnv
from IPython.display import clear_output
from numpy.random import random, choice

class QFrozenLakeAgent:

    def __init__(self, num_episodes=1000, max_steps=200, learning_rate=0.1,
                 gamma=0.9, epsilon=0.9, decay_rate=0.1,env=FrozenLakeEnv(map_name="10x10")):
        self.env = env
        state_size = self.env.observation_space.n
        actions_num = self.env.action_space.n
        self.q_table = np.zeros((state_size, actions_num))
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.avg_rewards = []


    def update_q_table(self, state, action):
        """
        Using Bellman equation updates Q Table with action, state
        and it's reward.
        """
        new_state, reward, done, _ = self.env.step(action)

        self.q_table[state, action] = self.q_table[
                                          state, action] + self.learning_rate \
                                      * (reward + self.gamma * np.max(
                                              self.q_table[new_state]) -
                                                  self.q_table[state, action])
        return new_state, reward, done

    def epsilon_greedy(self,state, Q, epsilon):

        values = Q[state, :]
        max_value = max(values)
        no_actions = len(values)

        greedy_actions = [a for a in range(no_actions) if values[a] == max_value]

        explore = (random() < epsilon)

        if explore:
            return choice([a for a in range(no_actions)])
        else:
            return choice([a for a in greedy_actions])

    def decay_epsilon(self, episode):
        """
        Decaying exploration with the number of episodes.
        """
        self.epsilon = 0.001 + 0.999 * np.exp(-self.decay_rate * episode)


    def train(self):
        """Training the agent to find the frisbee on the frozen lake"""

        self.avg_rewards = []
        self.episode_len = np.zeros(self.num_episodes)
        self.success = 0
        self.success_rate = []
        for episode in range(self.num_episodes):

            state = self.env.reset()

            for step in range(self.max_steps):
                action = self.epsilon_greedy(state,self.q_table,self.epsilon)
                state, reward, done = self.update_q_table(state, action)

                self.episode_len[episode] += 1

                if done:
                    if reward >= 0:
                        self.success += 1
                    break

            self.decay_epsilon(episode)
            if episode % 100 == 0 and episode > 0:
                self.success_rate.append(self.success / 100)
                self.success = 0


    def plot(self):
        """Plot the episode length and average rewards per episode"""

        fig = plt.figure(figsize=(20, 5))

        episode_len = [i for i in self.episode_len if i != 0]

        rolling_len = pd.DataFrame(episode_len).rolling(100, min_periods=100)
        mean_len = rolling_len.mean()
        std_len = rolling_len.std()

        plt.plot(mean_len, color='red')
        plt.fill_between(x=std_len.index, y1=(mean_len - std_len)[0],
                         y2=(mean_len + std_len)[0], color='red', alpha=.2)

        plt.ylabel('Episode length')
        plt.xlabel('Episode')
        plt.title(f'Frozen Lake - Length of episodes (mean over window size 100)')
        plt.show()
        plt.plot (range(len(self.success_rate)), self.success_rate,'r')
        plt.ylabel('Success_rate')
        plt.xlabel('Per 100 Episodes')
        plt.show()
        print(self.q_table)

def test(agent, num_episodes=5):
    """Let the agent play Frozen Lake"""

    time.sleep(2)

    for episode in range(num_episodes):
        state = agent.env.reset()
        done = False
        print('failed ', episode + 1)

        time.sleep(1.5)

        steps = 0
        while not done:
            clear_output(wait=True)
            agent.env.render()
            time.sleep(0.3)

            action = np.argmax(agent.q_table[state])
            state, reward, done, _ = agent.env.step(action)
            steps += 1

        clear_output(wait=True)
        agent.env.render()

        if reward == 1:
            print(f'Congratulation! 🏆 found in {steps} steps.')
            time.sleep(2)
        else:
            print('Sorry, you fell through a 🕳, try again!')
            time.sleep(2)
        clear_output(wait=True)



agent = QFrozenLakeAgent(env=FrozenLakeEnv(map_name="10x10") )
agent.train()
# test(agent, num_episodes=1)
agent.plot()
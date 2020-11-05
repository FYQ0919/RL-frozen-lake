

from itertools import product
from frozen_lake_env import FrozenLakeEnv
from IPython.display import clear_output
import time
import numpy as np
import matplotlib.pyplot as plt

def train(env,policy,epsilon=0.1):
    env.reset()
    finished = False
    episode_sar = []
    while not finished:
        current_s = env.s
        action = epsilon_action(policy[current_s],env,epsilon=epsilon)
        new_s, reward, finished, _info =  env.step(action)
        episode_sar.append([current_s,action,reward])
    #episode_sar.append([new_s,None,reward])
    return episode_sar

def epsilon_action(a,env,epsilon = 0.1):
    """
    Return the action most of the time but in 1-epsiolon of the cases a random action within the env.env.action_space is returned
    Return: action
    """
    rand_number = np.random.random()
    if rand_number < (1-epsilon):
        return a
    else:
        return env.action_space.sample()


    return episode_sar

def sar_to_sag(sar_list,GAMMA=0.96):

    G = 0
    state_action_gain = []
    for state,action,r in reversed(sar_list):
        G = r + GAMMA*G
        state_action_gain.append([state,action,G])
    return reversed(state_action_gain)


def monte_carlo(env, episodes=20000, epsilon=0.1):
    """
    Function for generating a policy the monte carlo way: Play a lot, find the optimal policy this way
    Args: env: the open ai gym enviroment object
    Return: policy: the "optimal" policy V: the value table for each s (optional)
    """
    #create a random policy
    policy = {j:np.random.choice(env.action_space.n) for j in range(env.observation_space.n)}
    #Gain or return is cummulative rewards over the entiere episode g(t) = r(t+1) + gamma*G(t+1)
    G = 0
    #Q function is essential for the policy update
    Q = {j:{i:0 for i in range(env.action_space.n)} for j in range(env.observation_space.n)}
    #The s,a pairs of the Q function are updated using mean of returns of each episode. So returns need to be collected
    returns = {(s,a):[] for s,a in product(range(env.observation_space.n),range(env.action_space.n))}
    success = 0
    for ii in range(episodes):

        seen_state_action_pairs = set()
        # convert S,A,R to S,A,G
        episode_sag = sar_to_sag(train(env,policy,epsilon=epsilon))
        #Use S,A,G to update Q (first-visit method), retruns and seen_state_action_paris
        for s,a,G in episode_sag:
            sa = (s, a)
            if sa not in seen_state_action_pairs:
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                seen_state_action_pairs.add(sa)
        # calculate new policy p[s] = argmax[a]{Q(s,a)}
        for s in policy.keys():
            policy[s] = max(Q[s],key=Q[s].get)
            _new_s, _reward, finished, _info = env.step(policy[env.s])
        if ii % 1000 == 0:
            print(f'episodes = {ii}')
    V = {s:max(list(Q[s].values())) for s in policy.keys()}
    return policy, V

def test_policy(env,policy,epsilon=0.1):
    print("Start Test")
    print(policy)
    env.reset()
    finished = False
    while not finished:
       clear_output(wait=True)
       env.render()
       time.sleep(0.3)
       current_s = env.s
       action = (policy[current_s])
       _new_s, _reward, finished, _info = env.step(action)
       clear_output(wait=True)
       env.render()
       # if _reward == 1:
       #     print("success")
       # else:
       #     print("false")

def main(env,map_size = 10,episodes = 10000,epsilon = 0.1):
    env.render()
    policy,V= monte_carlo(env,episodes=episodes,epsilon=epsilon)
    test_policy(env,policy,epsilon=0)
    for i in range(len(V)):
        V[i] = round(V[i],2)
    values = np.zeros((map_size,map_size),float)
    for i in range(map_size):
        for j in range(map_size):
          values[i,j] = (np.array(V[(i*map_size+j)]))

    fig, ax = plt.subplots()
    im = ax.imshow(values)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(map_size):
        for j in range(map_size):
            text = ax.text(j, i, values[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("MAP VALUES")
    fig.tight_layout()
    plt.show()
    print(V)

main(env = FrozenLakeEnv(map_name="4x4",is_slippery=True),map_size=4,episodes = 2000)





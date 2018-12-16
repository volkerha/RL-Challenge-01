import gym, time
import numpy as np
from getModel import getModelQube, getModelPendel
from gym.wrappers.monitor import Monitor
from sklearn.neural_network import MLPRegressor
from challenge1_template import get_model, get_policy
from scipy import spatial

env = Monitor(gym.make('Pendulum-v0'), 'training', video_callable=False,force=True)
env.seed(98251624)
max_num_samples = 10000
model = get_model(env, max_num_samples)


max_state = env.observation_space.high
min_state = env.observation_space.low
max_action = env.action_space.high
min_action = env.action_space.low
discret_states = 100
discrete_actions = 4
discount_factor = 0.99
theta = 1

def discreizeSpace(min_state,max_state, discret_num):
    discrete_space = []
    for i in range(0,len(max_state)):
        min = min_state[i]
        max = max_state[i]
        disc_temp = np.linspace(min, max, num=discret_num)
   
        discrete_space.append(disc_temp)
    discrete_space = np.array(discrete_space)
    return discrete_space.T


discrete_state_space = discreizeSpace(min_state,max_state, discret_states)
discrete_state_space_tree = spatial.KDTree(discrete_state_space)
discrete_action_space = discreizeSpace(min_action,max_action,discrete_actions)

def lookahead(state, V):
    A = np.zeros(discrete_actions)
    for action_idx in range(discrete_actions):
        obs = state
        act = discrete_action_space[action_idx]
        nobs_pred, rwd_pred = model(obs, act)
        distance, state_id = discrete_state_space_tree.query(nobs_pred)
        A[action_idx] += rwd_pred + discount_factor * V[state_id]   
    return A      

V = np.zeros(discret_states)
while True:
    delta = 0
    for state in discrete_state_space:
        A = lookahead(state, V)
        best_action_reward = np.max(A)
        distance, state_id = discrete_state_space_tree.query(state)
        delta = max(delta, np.abs(best_action_reward-V[state_id])) 
        V[state_id] = best_action_reward
    print(delta)    
    if delta < theta:
        break  

policy = np.zeros([discret_states, discrete_actions])  
for state in discrete_state_space:
    A = lookahead(state, V)
    best_action_id = np.argmax(A)
    distance, state_id = discrete_state_space_tree.query(state)
    policy[state_id, best_action_id] = 1.0


print(policy)        
    




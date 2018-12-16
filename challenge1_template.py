"""
Submission template for Programming Challenge 1: Dynamic Programming.
"""

info = dict(
    group_number=None,  # change if you are an existing seminar/project group
    authors="John Doe; Lorem Ipsum; Foo Bar",
    description="Explain what your code does and how. "
                "Keep this description short "
                "as it is not meant to be a replacement for docstrings "
                "but rather a quick summary to help the grader.")

import torch
import torch.nn as nn
import numpy as np
from scipy import spatial


def get_model(env, max_num_samples):
    """
    Sample up to max_num_samples transitions (s, a, s', r) from env
    and fit a parametric model s', r = f(s, a).

    :param env: gym.Env
    :param max_num_samples: maximum number of calls to env.step(a)
    :return: function f: s, a -> s', r
    """
    l1size = 256
    l2size = 128
    state_model = nn.Sequential(nn.Linear(4, l1size),
                          nn.ReLU(),
                          nn.Linear(l1size, l2size),
                          nn.ReLU(),
                          nn.Linear(l2size, env.observation_space.shape[0]))
    reward_model = nn.Sequential(nn.Linear(4, l1size),
                          nn.ReLU(),
                          nn.Linear(l1size, l2size),
                          nn.ReLU(),
                          nn.Linear(l2size, env.action_space.shape[0]))
    criterion = torch.nn.MSELoss()
    state_optimizer = torch.optim.Adam(state_model.parameters(), lr=0.005)
    reward_optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.005)

    # Sample some number of states
    replay_buffer = []
    s0 = env.reset()
    for i in range(max_num_samples):
        a = env.action_space.sample()
        s1, r, d, info = env.step(a)

        replay_buffer.append((s0, a, r, s1))

        if d:
            s0 = env.reset()
        else:
            s0 = s1

    # train model (s,a) -> (s',r)
    training_steps = 5000
    batch_size = 32
    for i in range(training_steps):
        sample_idxs = np.random.random_integers(0, len(replay_buffer)-1, batch_size)
        batch = [replay_buffer[s_i] for s_i in sample_idxs]
        batch = np.array(batch)
        s0 = [s0 for s0, a, r, s1 in batch]
        a = [a for s0, a, r, s1 in batch]
        r = [r for s0, a, r, s1 in batch]
        s1 = [s1 for s0, a, r, s1 in batch]

        s_in = torch.tensor(s0, dtype=torch.float32)
        a_in = torch.tensor(a, dtype=torch.float32) #.squeeze(1)
        input = torch.cat((s_in, a_in), 1)

        state_pred = state_model(input).squeeze(1)
        reward_pred = reward_model(input).squeeze(1)

        state_target = torch.tensor(s1, dtype=torch.float32)
        reward_target = torch.tensor(r, dtype=torch.float32)
        state_loss = criterion(state_pred, state_target)
        reward_loss = criterion(reward_pred, reward_target)

        state_optimizer.zero_grad()
        state_loss.backward()
        state_optimizer.step()

        reward_optimizer.zero_grad()
        reward_loss.backward()
        reward_optimizer.step()

    return lambda obs, act: (state_model(torch.cat((torch.tensor(obs, dtype=torch.float32), torch.tensor(act, dtype=torch.float32)))).detach().numpy(),
                            reward_model(torch.cat((torch.tensor(obs, dtype=torch.float32), torch.tensor(act, dtype=torch.float32)))).detach().numpy()[0])




def discreizeSpace(min_state,max_state, discret_num):
    discrete_space = []
    for i in range(0,len(max_state)):
        min = min_state[i]
        max = max_state[i]
        disc_temp = np.linspace(min, max, num=discret_num)
    
        discrete_space.append(disc_temp)
    discrete_space = np.array(discrete_space)
    return discrete_space.T

def value_iteration(model, observation_space, action_space):
    discret_states = 500
    discrete_actions = 3
    discount_factor = 0.99
    theta = 0.1
    max_state = observation_space.high
    min_state = observation_space.low
    max_action = action_space.high
    min_action = action_space.low

    


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


    return policy, discrete_state_space_tree, discrete_action_space


def get_policy(model, observation_space, action_space):
    """
    Perform dynamic programming and return the optimal policy.

    :param model: function f: s, a -> s', r
    :param observation_space: gym.Space
    :param action_space: gym.Space
    :return: function pi: s -> a
    """
    policy, discrete_state_space_tree,discrete_action_space  = value_iteration(model, observation_space, action_space)

    
    def state2action(obs):
        distance, state_id  = discrete_state_space_tree.query(obs)       
        action_id = np.argmax(policy[state_id])     
        #print('Actions: ', discrete_action_space,'Choosen Actionindex: ', action_id)   
        return discrete_action_space[action_id]
    
    return lambda obs: state2action(obs)

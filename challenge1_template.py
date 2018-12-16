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


def get_policy(model, observation_space, action_space):
    """
    Perform dynamic programming and return the optimal policy.

    :param model: function f: s, a -> s', r
    :param observation_space: gym.Space
    :param action_space: gym.Space
    :return: function pi: s -> a
    """
    return lambda obs: action_space.high

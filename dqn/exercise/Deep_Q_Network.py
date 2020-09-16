#!/usr/bin/env python
# coding: utf-8

# # Deep Q-Network (DQN)
# ---
# In this notebook, you will implement a DQN agent with OpenAI Gym's LunarLander-v2 environment.
# 
# ### 1. Import the Necessary Packages

# In[1]:
import pickle

import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# ### 2. Instantiate the Environment and Agent
# 
# Initialize the environment in the code cell below.

# In[2]:


env = gym.make('Breakout-v0')
env.seed(0)
print(f'State shape: {env.observation_space.shape} = {np.product(env.observation_space.shape)}')
print('Number of actions: ', env.action_space.n)

# In[3]:
from dqn_agent import Agent

agent = Agent(state_size=np.product(env.observation_space.shape), action_size=env.action_space.n, seed=0)

# watch an untrained agent
# state = env.reset()
# for j in range(200):
#     action = agent.act(state)
#     env.render()
#     state, reward, done, _ = env.step(action)
#     if done:
#         break
#
# env.close()


# ### 3. Train the Agent with DQN
# 
# Run the code cell below to train the agent from scratch.  You are welcome to amend the supplied values of the parameters in the function, to try to see if you can get better performance!
# 
# Alternatively, you can skip to the next step below (**4. Watch a Smart Agent!**), to load the saved model weights from a pre-trained agent.

# In[3]:

def plot_scores(scores, rolling_window=100):
    """Plot scores and optional rolling mean using specified window."""
    plt.plot(scores); plt.title("Scores");
    plt.xlabel('Episode #')
    plt.ylabel('Score')
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean);
    plt.savefig(f"scores_{pd.Timestamp.utcnow().value}.png")
    return rolling_mean


def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, save_path:str = None, save_every: int
= None,
        reload_path:str = None):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    if reload_path is not None:
        print("Reloading session...")
        agent.load(filepath=reload_path)
        with open(reload_path+"/scores.pickle", "rb") as f:
            scores =  pickle.load(f)

    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        actions_counter = {i: 0 for i in range(env.action_space.n)}
        for t in range(max_t):
            action = agent.act(state, eps)
            actions_counter[action] += 1
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}, Actions: {}, Last Score: {:.2f}'.format(i_episode,
                                                                                            np.mean(scores_window),
                                                                                            actions_counter,
                                                                                            score
                                                                                            ))
        # end=""
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break

        if save_every and save_path and i_episode % save_every == 0:
            print(f'\nSaving model to {save_path}')
            agent.save(filepath=save_path)
            with open(save_path + "/scores.pickle", "wb") as f:
                pickle.dump(scores, f)
            rolling_mean = plot_scores(scores)

    if save_path:
        agent.save(filepath=save_path)
        with open(save_path + "/scores.pickle", "wb") as f:
            pickle.dump(scores, f)

    return scores


scores = dqn(n_episodes=1000, save_every=100, save_path=".", reload_path='.')

rolling_mean = plot_scores(scores)

# ### 4. Watch a Smart Agent!
# 
# In the next code cell, you will load the trained weights from file to watch a smart agent!

# In[4]:


# load the weights from file
# agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
#
# for i in range(3):
#     state = env.reset()
#     for j in range(200):
#         action = agent.act(state)
#         env.render()
#         state, reward, done, _ = env.step(action)
#         if done:
#             break
#
# env.close()


# ### 5. Explore
# 
# In this exercise, you have implemented a DQN agent and demonstrated how to use it to solve an OpenAI Gym environment.  To continue your learning, you are encouraged to complete any (or all!) of the following tasks:
# - Amend the various hyperparameters and network architecture to see if you can get your agent to solve the environment faster.  Once you build intuition for the hyperparameters that work well with this environment, try solving a different OpenAI Gym task with discrete actions!
# - You may like to implement some improvements such as prioritized experience replay, Double DQN, or Dueling DQN! 
# - Write a blog post explaining the intuition behind the DQN algorithm and demonstrating how to use it to solve an RL environment of your choosing.  

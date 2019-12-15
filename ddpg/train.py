import os
import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

sys.path.append("../")

from agent import Agent

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print("GPU available: {}".format(torch.cuda.is_available()))
print("GPU tensor test: {}".format(torch.rand(3, 3).cuda()))

env = UnityEnvironment(file_name="../Reacher_Linux/Reacher.x86_64", no_graphics=False)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

state = env_info.vector_observations[0]  # get the current state
score = 0  # initialize the score

# Instantiate DDPG Agent
agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)


def ddpg(n_episodes=1000, max_t=1000):
    """Deep Deterministic Policy Gradient

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, True)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), '../checkpoints/checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), '../checkpoints/checkpoint_critic.pth')
            break
    return scores


scores = ddpg()

# plot the scores
fig = plt.figure()
# ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

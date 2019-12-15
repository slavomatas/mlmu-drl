import os
import numpy as np
import torch
from unityagents import UnityEnvironment

import sys

from agent import MADDPG
from utils import transpose_list, transpose_to_tensor

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available {}".format(torch.cuda.is_available()))

env = UnityEnvironment(file_name='/home/slavo/Dev//ma_collab-compet/Tennis_Linux/Tennis.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

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

agents = len(env_info.agents)

# how many steps before update
steps_per_update = 100

torch.set_num_threads(4)

# initialize policy and critic
maddpg_agent = MADDPG(state_size, action_size, agents)

for i, agent in enumerate(maddpg_agent.agents):
    agent.actor.load_state_dict(torch.load('./checkpoints/checkpoint_actor_' + str(i) + '.pth'))
    agent.critic.load_state_dict(torch.load('./checkpoints/checkpoint_critic_' + str(i) + '.pth'))


score = 0  # initialize the score

for i in range(1, 10000):  # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state (for each agent)
    scores = np.zeros(len(env_info.agents))  # initialize the score (for each agent)
    while True:
        actions = maddpg_agent.act(transpose_to_tensor(list(states)))

        actions = torch.stack(actions).view(-1).detach().cpu().numpy()
        actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]

        next_states = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished
        scores += env_info.rewards  # update the score (for each agent)
        states = next_states  # roll over states to next time step
        if np.any(dones):  # exit loop if episode finished
            break
    print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))

env.close()

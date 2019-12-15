import os
import torch
from unityagents import UnityEnvironment

import sys
sys.path.append("../")

from agent import Agent

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available {}".format(torch.cuda.is_available()))

env = UnityEnvironment(file_name="../Reacher_Linux/Reacher.x86_64", no_graphics=False)

# get the default brain# get t
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

# load the weights from file
agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)
agent.actor_local.load_state_dict(torch.load('../checkpoints/checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('../checkpoints/checkpoint_critic.pth'))

score = 0  # initialize the score

for i in range(3):
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    for j in range(4000):
        action = agent.act(state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        state = next_state
        score += reward
        print('\rScore: {:.2f}'.format(score), end="")
        if done:
            break

env.close()
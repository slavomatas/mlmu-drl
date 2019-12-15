import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from collections import deque
from agent import MADDPG
from buffer import ReplayBuffer
from utils import transpose_list, transpose_to_tensor
from unityagents import UnityEnvironment

BUFFER_SIZE = 100000  # replay buffer size
BATCH_SIZE = 1024 # minibatch size


def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train():
    seeding()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    print("GPU available: {}".format(torch.cuda.is_available()))
    print("GPU tensor test: {}".format(torch.rand(3, 3).cuda()))

    env = UnityEnvironment(file_name='/home/slavo/Dev//ma_collab-compet/Tennis_Linux/Tennis.x86_64', no_graphics=True)

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

    agents = len(env_info.agents)

    # number of training episodes.
    # change this to higher number to experiment. say 30000.
    number_of_episodes = 30000
    episode_length = 500

    # how many steps before update
    steps_per_update = 100

    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 1
    noise_reduction = 0.9999

    torch.set_num_threads(4)

    buffer = ReplayBuffer(BUFFER_SIZE)

    # initialize policy and critic
    maddpg_agent = MADDPG(state_size, action_size, agents)

    for i, agent in enumerate(maddpg_agent.agents):
        agent.actor.load_state_dict(torch.load('./checkpoints/checkpoint_actor_' + str(i) + '.pth'))
        agent.critic.load_state_dict(torch.load('./checkpoints/checkpoint_critic_' + str(i) + '.pth'))

    scores = []
    scores_window = deque(maxlen=100)  # last 100 scores

    actor_losses = []
    critic_losses = []
    for i in range(len(env_info.agents)):
        actor_losses.append([])
        critic_losses.append([])

    for episode in range(0, number_of_episodes):

        episode_rewards = []

        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        state_full = np.concatenate(state)

        # for calculating rewards for this particular episode - addition of all time steps
        for episode_t in range(episode_length+1):

            actions = maddpg_agent.act(transpose_to_tensor(list(state)), noise=noise)
            noise *= noise_reduction

            actions = torch.stack(actions).view(-1).detach().cpu().numpy()
            env_info = env.step(actions)[brain_name]

            state_next = env_info.vector_observations  # get the next state
            state_next_full = np.concatenate(state_next)
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done  # see if episode has finished

            # add experiences to buffer
            transition = (state, state_full, actions, rewards, state_next, state_next_full, dones)
            buffer.push(transition)

            episode_rewards.append(rewards)
            state, state_full = state_next, state_next_full

            # update once after every steps_per_update
            if len(buffer) > BATCH_SIZE and (episode_t > 0) and (episode_t % steps_per_update == 0):
                # print('maddpg update after {} steps'.format(episode_t))
                for agent_idx in range(len(env_info.agents)):
                    samples = buffer.sample(BATCH_SIZE)
                    al, cl = maddpg_agent.update(samples, agent_idx)
                    actor_losses[agent_idx].append(al)
                    critic_losses[agent_idx].append(cl)
                maddpg_agent.update_targets()  # soft update the target network towards the actual networks

        # calculate agent episode rewards
        agent_episode_rewards = []
        for i in range(len(env_info.agents)):
            agent_episode_reward = 0
            for step in episode_rewards:
                agent_episode_reward += step[i]
            agent_episode_rewards.append(agent_episode_reward)

        scores.append(np.max(agent_episode_rewards))
        scores_window.append(np.max(agent_episode_rewards))

        if episode > 10 and episode % 10 == 0:
            print('\rEpisode {}\tAgent Rewards [{:.4f}\t{:.4f}]\tMax Reward {:.4f}'.format(episode,
                                                                                           agent_episode_rewards[0],
                                                                                           agent_episode_rewards[1],
                                                                                           np.max(agent_episode_rewards)))

            print('\rEpisode {}\tAverage Actor 1 Loss {:.6f}\tAverage Critic 1 Loss {:.6f}'
                  '\tAverage Actor 2 Loss {:.6f}\tAverage Critic 2 Loss {:.6f}'.format(episode,
                                                                                       np.mean(actor_losses[0]),
                                                                                       np.mean(critic_losses[0]),
                                                                                       np.mean(actor_losses[1]),
                                                                                       np.mean(critic_losses[1])))

            print('\rEpisode {}\tAverage Score: {:.4f}'.format(episode, np.mean(scores_window)))

            # reset losses
            actor_losses = []
            critic_losses = []
            for i in range(len(env_info.agents)):
                actor_losses.append([])
                critic_losses.append([])

        if episode > 100 and episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.4f}'.format(episode, np.mean(scores_window)))

        if episode > 100 and np.mean(scores_window) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.4f}'.format(episode - 100,
                                                                                         np.mean(scores_window)))
            for i, save_agent in enumerate(maddpg_agent.agents):
                torch.save(save_agent.actor.state_dict(), './checkpoints/checkpoint_actor_'+str(i)+'.pth')
                torch.save(save_agent.critic.state_dict(), './checkpoints/checkpoint_critic_'+str(i)+'.pth')
            break

    env.close()
    return scores


if __name__ == '__main__':
    scores = train()

    # plot the scores
    fig = plt.figure()
    # ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

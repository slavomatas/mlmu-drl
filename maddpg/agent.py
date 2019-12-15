import torch
import torch.nn.functional as F

from model import Actor, Critic
from noise import OUNoise
from torch.optim import Adam
from utils import hard_update, soft_update, transpose_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    def __init__(self, state_size, action_size, n_agents, lr_actor=0.01, lr_critic=0.01):
        super(DDPGAgent, self).__init__()

        self.actor = Actor(state_size, action_size, seed=0).to(device)
        self.critic = Critic(state_size, action_size, n_agents).to(device)
        self.target_actor = Actor(state_size, action_size, seed=0).to(device)
        self.target_critic = Critic(state_size, action_size, n_agents).to(device)

        self.noise = OUNoise(action_size, scale=1.0)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1.e-5)

    def act(self, state, noise=0.0):
        '''
        state = state.to(device)
        action = self.actor(state) + (noise * self.noise.noise()).to(device)
        return action
        '''

        """Returns actions for given state as per current policy."""
        state = state.float().unsqueeze(0)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).squeeze(0)
        self.actor.train()

        action += (noise * self.noise.noise()).to(device)
        return torch.clamp(action, -1, 1)

    def target_act(self, state, noise=0.0):
        state = state.to(device)
        action = self.target_actor(state) + (noise * self.noise.noise()).to(device)
        return action


class MADDPG:
    def __init__(self, state_size, action_size, n_agents, discount_factor=0.95, tau=1e-3):
        super(MADDPG, self).__init__()
        self.action_size = action_size
        self.n_agents = n_agents
        self.agents = [DDPGAgent(state_size=state_size, action_size=action_size, n_agents=n_agents) for _ in
                       range(n_agents)]
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.agents, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = []

        for i in range(obs_all_agents.shape[1]):
            target_actions.append(self.agents[i].target_act(obs_all_agents[:, i, :], noise))

        return target_actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """

        states, states_all, actions_all, rewards, next_states, next_states_all, dones = [transpose_to_tensor(sample) for sample in samples]

        agent = self.agents[agent_number]
        agent.critic_optimizer.zero_grad()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network

        next_states = torch.stack(next_states)
        next_actions_all = self.target_act(next_states)
        next_actions_all = torch.cat(next_actions_all, dim=1)

        next_states_all = torch.stack(next_states_all)
        with torch.no_grad():
            q_targets_next = agent.target_critic(next_states_all, next_actions_all)

        # Compute Q targets for current states (y_i)
        q_targets = torch.stack(rewards)[:, agent_number].view(-1, 1) + self.discount_factor * q_targets_next * (
                    1 - torch.stack(dones)[:, agent_number].view(-1, 1))

        # Compute Q expected
        states_all = torch.stack(states_all)
        actions_all = torch.stack(actions_all)
        # actions_all = torch.stack(actions_all).view(-1, self.action_size * self.agents)
        q_expected = agent.critic(states_all, actions_all)

        # Compute critic loss

        # huber_loss = torch.nn.SmoothL1Loss()
        # critic_loss = huber_loss(current_Q, target_Q.detach())
        critic_loss = F.mse_loss(q_expected, q_targets.detach())

        # Minimize the loss
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        # update actor network using policy gradient
        agent.actor_optimizer.zero_grad()

        # ---------------------------- update actor ---------------------------- #
        actions_pred = []
        for i in range(self.n_agents):
            if i == agent_number:
                actions_pred.append(self.agents[i].actor(torch.stack(states)[:, i, :]))
            else:
                actions_pred.append(self.agents[i].actor(torch.stack(states)[:, i, :]).detach())

        actions_pred = torch.cat(actions_pred, dim=1)

        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already

        # get the policy gradient
        actor_loss = -agent.critic(states_all, actions_pred).mean()
        actor_loss.backward()

        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        return al, cl

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.agents:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)

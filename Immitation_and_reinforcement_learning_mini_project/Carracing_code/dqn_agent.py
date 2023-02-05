import numpy as np
import torch
import torch.optim as optim
from collections import namedtuple
from agent.replay_buffer import ReplayBuffer 

import pdb


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def tt(ndarray):
    return torch.autograd.Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)

class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, gamma=0.95, batch_size=64, epsilon=0.1, tau=0.01, lr=1e-4,
                 history_length=0):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            gamma: discount factor of future rewards.
            batch_size: Number of samples per batch.
            tau: indicates the speed of adjustment of the slowly updated target network.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
            lr: learning rate of the optimizer
        """
        # setup networks
#        self.Q = Q.cuda()
#        self.Q_target = Q_target.cuda()
#        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q = Q.cuda()
        self.Q_target = Q_target.cuda()
        self.Q_target.load_state_dict(self.Q.state_dict())
        
        # define replay buffer
        self.buffer_max_size = 1e5
        self.replay_buffer = ReplayBuffer(self.buffer_max_size)

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        
        self.epsilon_start = 1.0
        self.epsilon_end = 0.02
        self.epsilon_decay = 0.001
        self.current_step = 0

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr = lr)

        self.num_actions = num_actions

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """
        #Experience = namedtuple("Experience", ["states", "actions", "next_states", "rewards", "dones"])
        # TODO:
        # 1. add current transition to replay buffer
        #print(f"The reward is {reward} and action is {action} and state is {state}")
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)
        
        # 2. sample next batch and perform batch update:
        if self.replay_buffer.memory_check(self.batch_size):
            print("Entered")
            batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = self.replay_buffer.next_batch(self.batch_size)
            
            batch_states = tt(batch_states)
            batch_actions = tt(batch_actions) 
            batch_next_states = tt(batch_next_states)
            batch_rewards = tt(batch_rewards)
            batch_dones = tt(batch_dones)
        
        #       2.1 compute td targets and loss 
        #              td_target =  reward + discount * max_a Q_target(next_state_batch, a)
            with torch.no_grad():
                td_target = batch_rewards.cuda() + (1 - batch_dones.cuda())*self.gamma*torch.max(self.Q_target(batch_next_states.cuda()), dim = -1)[0]
        #       2.2 update the Q network

            #print("############################################################################################")
            #pdb.set_trace()
            current_prediction = self.Q(batch_states.cuda())[torch.arange(self.batch_size).long(), batch_actions.long()]
            loss = self.loss_function(current_prediction, td_target.detach())
        
        #loss_val = loss.detach().item()
       # print(f"The loss is: {loss_val}")
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        #       2.3 call soft update for target network
        #           soft_update(self.Q_target, self.Q, self.tau)
            soft_update(self.Q_target, self.Q, self.tau)

    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.current_step * self.epsilon_decay)
        self.current_step += 1
        r = np.random.uniform()

        if deterministic or r > self.epsilon:
            #pass
            # TODO: take greedy action (argmax)
            #action_id = np.argmax(self.Q(state).detach().numpy())
            state=tt(state)
            state = state.cuda()
            with torch.no_grad():           
                q_values = self.Q(state.unsqueeze(0))
            max_q_value = torch.argmax(q_values, dim=1)[0]
            action_id = max_q_value.cpu().detach().numpy()
        else:
            #pass
            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work. 
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            #action_id = np.random.choice(self.num_actions, p=[0.01, 0.03, 0.09, 0.86, 0.01])
            action_id = np.random.choice(self.num_actions, p=[0.05, 0.15, 0.15, 0.6, 0.05])
            
        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))

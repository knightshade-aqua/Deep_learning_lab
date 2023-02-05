import sys
import os
sys.path.append("../") 

import numpy as np
import gym
import itertools as it
import pdb
from agent.dqn_agent import DQNAgent
from tensorboard_evaluation import *
from agent.networks import MLP
from utils import EpisodeStats


def run_episode(env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    
    stats = EpisodeStats()        # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    while True:
        
        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:  
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if rendering:
            env.render()

        if terminal or step > max_timesteps: 
            break

        step += 1

    return stats

def train_online(env, agent, num_episodes, eval_cycle, num_eval_episodes, model_dir="./models_cartpole", tensorboard_dir="./tensorboard"):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")

    tensorboard = Evaluation(tensorboard_dir, "Logging", stats=["episode_reward", "a_0", "a_1", "mean_episode_reward"])

    # training
    for i in range(num_episodes):
        print("episode: ",i)
        stats = run_episode(env, agent, deterministic=False, do_training=True, rendering=False)
        #print("Episode reward is {stats.episode_reward}")
        tensorboard.write_episode_data(i, eval_dict={  "episode_reward" : stats.episode_reward, 
                                                                "a_0" : stats.get_action_usage(0),
                                                                "a_1" : stats.get_action_usage(1)})

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to 
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        if i % eval_cycle == 0:
            #eval_stats = EpisodeStats()
            eval_stats_mean_reward = 0
            for j in range(num_eval_episodes):
                eval_stats = run_episode(env, agent, deterministic=True, do_training=False)
                
            eval_stats_mean_reward = (eval_stats.episode_reward / num_eval_episodes)
            tensorboard.write_episode_data(i, eval_dict={  "mean_episode_reward" : eval_stats_mean_reward})
            print(f"Mean Episode Reward for eval_cycle {i}: {eval_stats_mean_reward} ")
            
        # store model.
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent.pt"))
   
    tensorboard.close_session()


if __name__ == "__main__":

    num_eval_episodes = 5   # evaluate on 5 episodes
    eval_cycle = 20         # evaluate every 10 episodes
    num_episodes = 400

    # You find information about cartpole in 
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

    env = gym.make("CartPole-v0").unwrapped

    state_dim = 4
    num_actions = 2

    # TODO: 
    # 1. init Q network and target network (see dqn/networks.py)
    Q = MLP(state_dim, num_actions)
    #pdb.set_trace()
    Q_target = MLP(state_dim, num_actions)
    # 2. init DQNAgent (see dqn/dqn_agent.py)
    dqn_agent = DQNAgent(Q, Q_target, num_actions)
    # 3. train DQN agent with train_online(...)
    train_online(env, dqn_agent, num_episodes, eval_cycle, num_eval_episodes)
 

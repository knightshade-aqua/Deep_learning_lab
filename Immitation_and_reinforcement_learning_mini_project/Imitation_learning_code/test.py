from __future__ import print_function

import sys
sys.path.append("../") 

from datetime import datetime
import numpy as np
import gym
import os
import json
import torch
import pdb

from agent.bc_agent import BCAgent
from utils import *


def run_episode(env, agent, rendering=True, max_timesteps=1000, N=5):
    
    episode_reward = 0
    step = 0
    

    image_hist = []
    state = env.reset()
    #print(state.shape)
    state = rgb2gray(state)
    image_hist.extend([state] * N)
    state = np.array(image_hist).reshape(1, N, 96, 96)
    state = torch.tensor(state, dtype = torch.float32)
    
    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events() 

    while True:
        
        # TODO: preprocess the state in the same way than in your preprocessing in train_agent.py
        #state = rgb2gray(state).reshape(1,1,96,96)
	

    	
        #pdb.set_trace()
        #state.cpu()
        #print(state.shape)

        
        # TODO: get the action from your agent! You need to transform the discretized actions to continuous
        # actions.
        # hints:
        #       - the action array fed into env.step() needs to have a shape like np.array([0.0, 0.0, 0.0])
        #       - just in case your agent misses the first turn because it is too fast: you are allowed to clip the acceleration in test_agent.py
        #       - you can use the softmax output to calculate the amount of lateral acceleration
        # a = ...
        a = agent.predict(state).detach().numpy()
        a = np.argmax(a)
        print(a)
        a = id_to_action(a)
        print(a)
        next_state, r, done, info = env.step(a)   
        episode_reward += r       
        
        next_state = rgb2gray(next_state)
        image_hist.pop(0)
        image_hist.append(next_state)
        next_state = np.array(image_hist).reshape(1, N, 96, 96)
        state = next_state
        state = torch.tensor(state, dtype = torch.float32)
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True              
    
    n_test_episodes = 15                  # number of episodes to test
    N = 5
    # TODO: load agent
    agent = BCAgent()
    agent.load("models/agent_history_5.pt")
    #my_model = net.load_state_dict(torch.load('classifier.pt', map_location=torch.device('cpu')))

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering, N=5)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
    
 
    fname = "results/results_bc_agent_1-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

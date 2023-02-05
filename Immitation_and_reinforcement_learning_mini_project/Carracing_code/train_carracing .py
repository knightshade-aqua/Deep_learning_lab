# export DISPLAY=:0 

import sys
sys.path.append("../") 
import pdb
import numpy as np
import gym
from tensorboard_evaluation import *
from utils import EpisodeStats, rgb2gray
from utils import *
from agent.dqn_agent import DQNAgent
from agent.networks_cnn import CNN
import os

""" def change_max_time_steps(var, max_time_steps):
    if var < int(max_time_steps/100):
        time_steps = int(max_time_steps/10)
    elif var < int(max_time_steps/25):
        time_steps = int(max_time_steps/25)
    else:
        time_steps = max_time_steps
    
    return time_steps """
def change_max_time_steps(var):
#    time_steps_temp = int(2*var + 150)
#    time_steps = min(time_steps_temp, max_time_steps) 
#    return time_steps

    if var <= 5:
        time_steps = 2*var + 20
    elif var > 5 and var <=10 :
        time_steps = 2*var + 30
    elif var > 10 and var <= 30:
        time_steps = 2*var + 50
    elif var > 30 and var <= 50:
        time_steps = 2*var + 100
    elif var > 50 and var <= 150:
        time_steps = 2*var + 500
    else:
        time_steps = 1000
    
    return time_steps
        
def your_id_to_action_method(action_id, max_speed=0.9):
    """ 
    this method makes actions continous.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    #a = np.array([0.0, 0.0, 0.0])
    #LEFT = 1, RIGHT = 2, ACCELERATE = 3, BRAKE = 4, STRAIGHT = 0
    if action_id == 1:
        return np.array([-1.0, 0.0, 0.05])
    elif action_id == 2:
        return np.array([1.0, 0.0, 0.05])
    elif action_id == 3:
        return np.array([0.0, max_speed, 0.0])
    elif action_id == 4:
        return np.array([0.0, 0.0, 0.1])
    else:
        return np.array([0.0, 0.2, 0.0])

def run_episode(env, agent, max_timesteps, deterministic, skip_frames=4,  do_training=True, rendering=False, history_length=0):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events() 

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(history_length + 1, 96, 96)
    #pdb.set_trace()
    #state = np.array(state).reshape(1,96,96)

    #state = np.array(state).reshape(-1,1,96,96)
    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly. 
        action_id = agent.act(state, deterministic)
        action = your_id_to_action_method(action_id)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r
            #print(terminal)

            if rendering:
                env.render()

            if terminal: 
                 break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(next_state).reshape(history_length + 1, 96, 96,)   ## review this

        if do_training:
            print("I am in training")
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        val_num = (step * (skip_frames + 1))
        

        if terminal or (step * (skip_frames + 1)) > max_timesteps : 
            print("Code escaped")
            print(f"Terminal is {terminal}, Max steps is {max_timesteps}, condition is {val_num}")
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, history_length=0, model_dir="./models_carracing", tensorboard_dir="./tensorboard"):
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")
    
    tensorboard = Evaluation(tensorboard_dir, "train", ["episode_reward", "straight", "left", "right", "accel", "brake","mean_episode_rewards"])
    
    best_mean_reward_temp = 50
    
    epir = []
    mepir = []
    ac = []
    brake = []
    left = []
    right = []
    straight = []
    
    for i in range(num_episodes):
        print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
        var_timesteps = change_max_time_steps(i)
        
        stats = run_episode(env, agent, max_timesteps=var_timesteps, deterministic=False, do_training=True)
        #print("################Episode ended################")

        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward, 
                                                      "straight" : stats.get_action_usage(STRAIGHT),
                                                      "left" : stats.get_action_usage(LEFT),
                                                      "right" : stats.get_action_usage(RIGHT),
                                                      "accel" : stats.get_action_usage(ACCELERATE),
                                                      "brake" : stats.get_action_usage(BRAKE), })
        epir.append(stats.episode_reward)
        right.append(stats.get_action_usage(RIGHT))
        left.append(stats.get_action_usage(LEFT))
        brake.append(stats.get_action_usage(BRAKE))
        ac.append(stats.get_action_usage(ACCELERATE))
        straight.append(stats.get_action_usage(STRAIGHT))
        print("++++++++++++++++Data+++++++++++++++++++++")
        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to 
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        if i % eval_cycle == 0:
            eval_stats_mean_reward = 0
            for j in range(num_eval_episodes):
                eval_stats = run_episode(env, agent, max_timesteps=var_timesteps, deterministic=True, do_training=False)
            eval_stats_mean_reward = (eval_stats.episode_reward / num_eval_episodes)
            
            tensorboard.write_episode_data(i, eval_dict={ "mean_episode_rewards" : eval_stats_mean_reward})
            mepir.append(eval_stats_mean_reward)
            
        print(f"Mean reward is: {eval_stats.episode_reward}")
        print(f"The reward is: {stats.episode_reward}")
        
        # store model.
        if best_mean_reward_temp < eval_stats_mean_reward:
            best_mean_reward_temp = eval_stats_mean_reward
            agent.save(os.path.join(model_dir, "dqn_agent.pt"))
        if i % eval_cycle == 0 or (i >= num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent.ckpt")) 
    
    agent.save(os.path.join(model_dir, "final_dqn_agent.pt"))
    with open('epir.txt', 'w') as f:
        for item in epir:
            f.write("%s\n" % item)
    f.close()
    
    with open('mepir.txt', 'w') as f:
        for item in mepir:
            f.write("%s\n" % item)
    f.close()        
    with open('ac.txt', 'w') as f:
        for item in ac:
            f.write("%s\n" % item)
    f.close()        
    with open('brake.txt', 'w') as f:
        for item in brake:
            f.write("%s\n" % item)
    f.close()        
    with open('left.txt', 'w') as f:
        for item in left:
            f.write("%s\n" % item)
    f.close()        
    with open('right.txt', 'w') as f:
        for item in right:
            f.write("%s\n" % item)
    f.close()
    with open('straight.txt', 'w') as f:
        for item in straight:
            f.write("%s\n" % item)
    f.close()
    tensorboard.close_session()

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0

if __name__ == "__main__":

    num_eval_episodes = 5
    eval_cycle = 20
    num_actions = 5

    env = gym.make('CarRacing-v0').unwrapped

    # TODO: Define Q network, target network and DQN agent
    Q = CNN()
    #pdb.set_trace()
    Q_target = CNN()
    # 2. init DQNAgent (see dqn/dqn_agent.py)
    agent_net = DQNAgent(Q, Q_target, num_actions)
    
    Q_target.load_state_dict(Q.state_dict())
    
    train_online(env, agent_net, num_episodes=400, history_length=0, model_dir="./models_carracing")


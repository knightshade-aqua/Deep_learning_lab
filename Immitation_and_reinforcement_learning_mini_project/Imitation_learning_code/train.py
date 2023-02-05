from __future__ import print_function

import sys
sys.path.append("../") 

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import torch

from utils import *
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation
from torch.utils.data import DataLoader,TensorDataset, WeightedRandomSampler
#from collections import Counter

def frame_stacking(X_var, history_length):
    N = history_length
    x_temp = []
    x_store = np.zeros((N,96,96))
    x_var_temp = np.zeros((X_var.shape[0],N,96,96))
    for i in range(X_var.shape[0]):
        if i == 0:
            x_temp.extend([X_var[i,:,:,:]]*(N))
            x_store = np.array(x_temp).reshape(N, 96, 96)
            x_var_temp[i,:,:,:] = x_store
        else:
            x_temp.pop(0)
            x_temp.append(X_var[i,:,:,:])
            x_store = np.array(x_temp).reshape(N, 96, 96)
            x_var_temp[i,:,:,:] = x_store
    
    return x_var_temp

def sampling(data):
    n = data.shape[0]
    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    
    for i in data:
        if i==0:
            count_0 = count_0 + 1
        elif i==1:
            count_1 = count_1 + 1
        elif i==2:
            count_2 = count_2 + 1
        else:
            count_3 = count_3 + 1
   # print(f"0 is {count_0}, 1 is {count_1}, 2 is {count_2}, 3 is {count_3} ")
    
    class_weights = [(n/(4*count_0)), (n/(4*count_1)), (n/(4*count_2)), (n/(4*count_3))]
    return class_weights

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    X_train = np.array([(rgb2gray(image)/255.0).reshape(1,96,96) for image in X_train])
    X_valid = np.array([(rgb2gray(image)/255.0).reshape(1,96,96) for image in X_valid])
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space 
    #    using action_to_id() from utils.py.
    y_valid = np.array([action_to_id(action) for action in y_valid])
    y_train = np.array([action_to_id(action) for action in y_train])
    
    X_train = frame_stacking(X_train, history_length = 1)
    X_valid = frame_stacking(X_valid, history_length = 1)
    
    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    
    return X_train, y_train, X_valid, y_valid


def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")


    # TODO: specify your agent with the neural network in agents/bc_agent.py 
    agent = BCAgent()
    
    tensorboard_eval = Evaluation(tensorboard_dir,"Metrics",["Training Loss","Validation Loss","Training Accuracy","Validation Accuracy"])

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.int64)
    
    train_data = TensorDataset(X_train, y_train)
    
    class_weights = sampling(y_train)
    sample_weights = [0]*len(train_data)

    
    for idx, (data,label) in enumerate(train_data):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight
    sampler = WeightedRandomSampler(sample_weights, num_samples = len(sample_weights), replacement=True)   
    train_loader = DataLoader(train_data,batch_size=batch_size, sampler = sampler,drop_last=True)
    
    test_data = TensorDataset(X_valid, y_valid)
    class_weights_val = sampling(y_valid)
    sample_weights_val = [0]*len(test_data)
    
    for idx, (data,label) in enumerate(test_data):
        class_weight_val = class_weights_val[label]
        sample_weights_val[idx] = class_weight_val
    sampler_val = WeightedRandomSampler(sample_weights_val, num_samples = len(sample_weights_val), replacement=True)   
    test_loader = DataLoader(test_data,batch_size=batch_size, sampler = sampler_val,drop_last=True)
   
    train_acc = 0
    val_acc = 0
    train_loss = 0
    
    #val_loss_func = torch.nn.CrossEntropyLoss() 
    
    for i in range(n_minibatches):
        batch_acc  = []
        batch_loss = []
        
        counter = 1
        for X,y in train_loader:
            print(f"Epoch: {i+1}, Batch: {counter}")
            X,y = X.cuda(), y.cuda()
            loss, outputs = agent.update(X,y)
            batch_acc.append( 100*torch.mean((torch.argmax(outputs,axis=1) == y).float()).item())
            batch_loss.append(loss.item())
            counter += 1

        train_acc = np.mean(batch_acc)
        train_loss = np.mean(batch_loss)
        
        
            X,y = next(iter(test_loader)) # extract X,y from test dataloader
            out = agent.predict(X)
            predlabels = torch.argmax( out,axis=1 )
            #predlabels = torch.argmax( agent.predict(X),axis=1 )
            val_acc = 100*torch.mean((predlabels == y.cuda()).float()).item()
            
            
        
        print(f"Train Loss: {train_loss}, Validation Loss: {val_loss}, Train Accuracy: {train_acc}, Validation Accuracy: {val_acc}")
        
        tensorboard_eval.write_episode_data(i,eval_dict = {"Training Loss" : train_loss, "Validation Loss": val_loss, "Training Accuracy" : train_acc, "Validation Accuracy" : val_acc})
      
    # TODO: save your agent
    model_dir = agent.save(os.path.join(model_dir, "agent_history_1.pt"))
    print("Model saved in file: %s" % model_dir)


if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=3)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, n_minibatches=50, batch_size=64, lr=1e-4)
 

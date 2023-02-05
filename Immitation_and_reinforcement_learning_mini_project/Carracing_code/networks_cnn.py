import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""

class CNN(nn.Module):

    def __init__(self, history_length=0, n_classes=5): 
        super(CNN, self).__init__()
        # TODO : define layers of a convolutional neural network
        self.conv1 = nn.Conv2d((history_length + 1), 64, kernel_size=3)
        self.batch_norm_1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3)
        self.batch_norm_2 = nn.BatchNorm2d(32)
        
        #self.conv2 = nn.Conv2d(128, 64, kernel_size=3)
        #self.batch_norm_2 = nn.BatchNorm2d(64)
        
        self.max_pool = nn.MaxPool2d(4)
        
        self.flatten_layer = nn.Flatten()
        
        self.fc1 = nn.Linear(23*23*32,128)
        #self.batch_norm_3 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 64)
        #self.batch_norm_4 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64,n_classes)
        
        self.drop = nn.Dropout(p=0.5)
    

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch_norm_1(x)
        
        
        x = F.relu(self.conv2(x))
        x = self.batch_norm_2(x)
        
        
        #x = F.relu(self.conv3(x))
        #x = self.batch_norm_3(x)
        
        
        x = self.max_pool(x)
        
        x = self.flatten_layer(x)
        
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        
        x = F.relu(self.fc2(x))
        x = self.drop(x)       
        x = self.fc3(x)
        
        return x


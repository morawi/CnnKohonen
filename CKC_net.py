# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 20:14:31 2021

@author: malrawi



Details about the models are below:
https://github.com/lukemelas/EfficientNet-PyTorch


Name	          #Params	Top-1-Acc.	Pretrained
----------------------------------------------------
efficientnet-b0	    5.3M	 76.3	        ✓
efficientnet-b1	    7.8M	 78.8	        ✓
efficientnet-b2	    9.2M	 79.8	        ✓
efficientnet-b3	    12M	     81.1	        ✓
efficientnet-b4	    19M	     82.6	        ✓
efficientnet-b5	    30M	     83.3	        ✓
efficientnet-b6	    43M	     84.0	        ✓
efficientnet-b7	    66M	     84.4	        ✓
----------------------------------------------------

There is also a new, large efficientnet-b8 pretrained model that is only available in advprop form. When using these models, replace ImageNet preprocessing code as follows:

if advprop:  # for models using advprop pretrained weights
    normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
else:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])




"""

import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet
import numpy as np


def get_winners(output_shape, winners_idx=None, lambda_val=1, neighbor_idx=2):
    # usage:
    # zz =  get_winners([4, 100], winners_idx = np.array([50, 40, 70, 80]), neighbor_idx=10)
    
     # i= torch.tensor([[0, 1, 1], [2, 0, 2]])
     # v =v = torch.tensor([3, 4, 5], dtype=torch.float32)
    # zz = torch.sparse_coo_tensor(i, v, [5, 8]).to_dense()
    # tensor([[0., 0., 3., 0., 0., 0., 0., 0.],
    #     [4., 0., 5., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0.]])
    
    num_samples = len(winners_idx) # num_samples = output_shape[0]
    x_idx = torch.arange(0, num_samples)
    winner_nodes = torch.zeros(output_shape)
    if neighbor_idx>0: 
        for n_idx in range(1, neighbor_idx, 1):
            winners_next = winners_idx + n_idx*(winners_idx<(output_shape[1]-n_idx)) # (winners_idx+2)%output_shape[1] # % used to keep the values within the upper bound
            winners_prev = winners_idx - n_idx 
            winners_prev *= winners_prev>0 # in case we have a -ve value, set them to zero     # a *= (a>0)   # https://stackoverflow.com/questions/3391843/how-to-transform-negative-elements-to-zero-without-a-loop
            # now, setting amplitiudes of the neigbors and the max node
            amp = 1 - np.exp(-4/n_idx)
            winner_nodes[x_idx, winners_next] = amp
            winner_nodes[x_idx, winners_prev] = amp
    winner_nodes[x_idx, winners_idx] = 1 # 100% power to the winner node
    
    # we can use some fft low pass filter to smooth the winners 
    
    return winner_nodes



# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/


class FeedforwardNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNet, self).__init__()
        # Linear function 1: 784 --> 100
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity 1
        self.relu1 = nn.ReLU()

        # Linear function 2: 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()

        # Linear function 3: 
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 3
        self.relu3 = nn.ReLU()

        # Linear function 4 (readout): 
        self.fc4 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)

        # Linear function 2
        out = self.fc3(out)
        # Non-linearity 2
        out = self.relu3(out)

        # Linear function 4 (readout)
        out = self.fc4(out)
        return out



class CKCNet(nn.Module):

    def __init__(self, num_classes, 
                 num_epochs, max_num_neighbors,
                 kohonen_dim=128, # this shuld be larger than num_classes
                 hidden_units=64,
                 model_name='efficientnet-b0',                  
                 pretrained = True
                 ):
        super().__init__()
        
        assert(num_classes<kohonen_dim)        
        self.neighbors_map = [1+max_num_neighbors*(num_epochs-1-epoch)//num_epochs for epoch in range(0, num_epochs)]
        # self.neighbors_map = [1+kohonen_dim*(num_epochs-1-epoch)//num_epochs for epoch in range(0, num_epochs)]
        self.winner_nodes = None
        self.epoch_reached= 0
        self.max_num_neighbors = max_num_neighbors
        
        # EfficientNet    
        if pretrained:
            # self.network1 = EfficientNet.from_pretrained(model_name, 
            #                  num_classes = kohonen_dim, include_top=True) # in_channels=1)
            self.network1 = EfficientNet.from_pretrained(model_name, 
                                                         include_top=True) # in_channels=1)
            # model = EfficientNet.from_pretrained("efficientnet-b0", advprop=True) # what's the difference?
        else:
            # self.network1 = EfficientNet.from_name(model_name, 
            #                  num_classes= kohonen_dim, include_top=True) # in_channels=1)            
            self.network1 = EfficientNet.from_pretrained(model_name, 
                             include_top=True) # in_channels=1)
        
        self.network2 = FeedforwardNet(kohonen_dim, hidden_dim = hidden_units, output_dim = num_classes)
        
                         
    def forward(self, x1, epoch=None):
        
        if epoch is not None:
            self.epoch_reached = epoch
            
        out = self.network1(x1)                
        self.winner_nodes = get_winners(out.shape, torch.argmax(out, dim=1), lambda_val=1,                                    
                                   neighbor_idx=self.neighbors_map[self.epoch_reached])                    
        out = self.winner_nodes.cuda()*out
        out = self.network2(out) 
        
        return out
    





# https://github.com/lukemelas/EfficientNet-PyTorch/pull/208
# model = EfficientNet.from_name("efficientnet-b0", num_classes=2, include_top=True, in_channels=1)
# #self.network1 = EfficientNet.from_pretrained(model_name)        
        # # Replace last layer
        # self.network1._fc = nn.Sequential(nn.Linear(self.network1._fc.in_features, 512), 
        #                                  nn.ReLU(),  
        #                                  nn.Dropout(0.25),
        #                                  nn.Linear(512, 128), 
        #                                  nn.ReLU(),  
        #                                  nn.Dropout(0.50), 
        #                                  nn.Linear(128, num_classes))    
    
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:40:34 2021

@author: malrawi
"""
import torch
import numpy as np

def get_winners(output, neighbor_radius=10):
    # usage:
    # output=torch.rand([4,100]); zz =  get_winners(output,  neighbor_radius=30)
    
    output_shape = output.shape
    winners_idx= torch.argmax(output, dim=1)
        
    num_samples = len(winners_idx) # num_samples = output_shape[0]
    x_idx = torch.arange(0, num_samples)
    winner_nodes = torch.zeros(output_shape)
    if neighbor_radius>0: 
        for n_idx in range(1, neighbor_radius, 1):
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

# output=torch.rand([4,100]); zz =  get_winners(output,  neighbor_radius=30)


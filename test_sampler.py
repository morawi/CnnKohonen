# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:17:20 2021

@author: malrawi

"""
import torch
from torch.utils.data import SubsetRandomSampler as SubSetRandSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

my_sampler_size = 5000

 
val_transforms = transforms.Compose([
        # transforms.Resize(image_size, interpolation=Image.BICUBIC),
        # transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        # normalize,
    ])

val_dataset = datasets.CIFAR10('../data', train= False, download=True,
                        transform = val_transforms, 
                        )                                

validate_significance_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=100,         
        num_workers=0, 
        # sampler = my_sampaler,
        sampler = SubSetRandSampler(torch.randint(0, len(val_dataset), (my_sampler_size,)) ), # SubSetRandSampler(range(1000)),
        shuffle= False, 
        pin_memory=True)

def get_loader(dataset):
    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=100,         
            num_workers=0, 
            # sampler = my_sampaler,
            sampler = SubSetRandSampler(torch.randint(0, len(val_dataset), (my_sampler_size,)) ), # SubSetRandSampler(range(1000)),
            shuffle= False, 
            pin_memory=True)
    return loader



    

'''unit-test'''
val_loader = get_loader(val_dataset)
for jj in range(5):    
    print('Iteration:', jj)
    for ii, (images, target) in enumerate(val_loader):        
        if ii<2:
            plt.imshow(images[ii,:].permute(1, 2, 0)  ); plt.show() # to see the image
            print(target[ii].item())
            
            break


    
    


# def get_the_sampler(posterior_prob, sampler_size, labels, no_classes): 
#     ''' 
#     Args in - 
#         score: posteriori values for each of the labels [0 to 1], 1 indicates 
#         the label has high likliness to be of a correct class
#         sampler_sie: the intended sampler size the user wants to get back, the 
#         size of the returned sampler will be sligthly less than this
#         labels: an array containing the labels
#         no_classes: the number of classes in the problem        
        
#     Parameters - 
#         percentage_of_selected_samples: selecting 50% for the samples with the highest 
#         'score' values 
#     '''
    
#     percentage_of_selected_samples = 50/100
    
#     len_labels_per_class = np.zeros(no_classes, dtype=int)      
#     idx_per_class = np.zeros([no_classes, len(labels)], dtype=int)
#     for i in range(no_classes):
#         idx_per_class[i] = labels==i
#         len_labels_per_class[i] = sum(idx_per_class[i] == True)
#     no_labels_per_class = min(len_labels_per_class)   
#     sampler_pool_size = int( no_labels_per_class * percentage_of_selected_samples ) 
    
#     sampler_size = int(sampler_size/no_classes)
#     if(sampler_size > sampler_pool_size): 
#         print('Probably you need to decrease the value percentage_of_selected_samples: ', percentage_of_selected_samples)
#         exit('Exiting function get_the_sampler(): sampler_size has become larger than sampler_pool_size')
        
    
#     my_sampler = []
#     for i in range(no_classes):
#         sample_idx = (-posterior_prob[idx_per_class[i]]).argsort()[:sampler_pool_size]   
#         sample_idx = np.random.permutation(sample_idx)
#         sample_idx = sample_idx[:sampler_size]
#         my_sampler.extend(sample_idx)
    
#     if len(my_sampler) <100:  exit('Exiting function get_the_sampler(): small sampler')    
#     my_sampler = torch.utils.data.sampler.SubsetRandomSampler(my_sampler)  
#     return my_sampler

# # my_sampler = get_the_sampler(posterior_prob, my_sampler_size, pred_labels, len(classes)) 



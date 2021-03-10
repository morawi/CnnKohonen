# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 19:17:26 2021

@author: malrawi
"""
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
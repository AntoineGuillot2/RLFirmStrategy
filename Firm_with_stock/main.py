#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:35:25 2018

@author: antoine
"""
from envir import Environment
import numpy as np
import matplotlib.pyplot as plt
N_TURN=100
N_SIMU=300

###Environment parameters
environment_params={}
environment_params['price_evolution']={'average':0,'seasonal':10,'period':24,'random':2,'trend':0.01}
environment_params['n_firm']=1

###Firm parameters
firm_params={}
firm_params['initial_funds']=1000
firm_params['replay_memory_size']=1000
firm_params['memory_size']=12
firm_params['plot_frequency']=10

max_prod=20
max_prod+=1
max_sell=20
max_sell+=1
firm_params['possible_actions']=np.concatenate(([np.array(range(max_prod*max_sell))%max_sell],[np.array(range(max_prod*max_sell))%max_prod]),0).transpose()


firm_params['cost']=lambda x: 0.5*(x**2)
firm_params['max_stock']=10
firm_params['verbose']=0
firm_params['initial_explore_rate']=1
firm_params['explore_rate_decay']=0.98
firm_params['min_explore_rate']=0.05
firm_params['explore_turns']=290
firm_params['WACC']=0.01
firm_params['production_time']=4
firm_params['initial_inventory']=10
firm_params['max_inventory']=20
firm_params['epochs']=10
firm_params['epsilon_greedy']=False

if __name__ == '__main__':
    envir_glob=Environment(environment_params,firm_params)
    for simulation in range(N_SIMU):
        opt_prod_no_noise=[]
        for turn in range(N_TURN):
            envir_glob.act()
        envir_glob.reset()
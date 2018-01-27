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
N_SIMU=50

###Environment parameters
environment_params={}
environment_params['price_evolution']={'average':2,'seasonal':6,'period':12,'random':0}
environment_params['n_firm']=1


###Firm parameters
firm_params={}
firm_params['WACC']=0.01
firm_params['initial_funds']=1000
firm_params['replay_memory_size']=2000
firm_params['memory_size']=12
firm_params['plot_frequency']=1
###Production and sells
max_prod=10
max_prod+=1
max_sell=20
max_sell+=1
firm_params['possible_actions']=np.concatenate(([np.array(range(max_prod*max_sell))%max_prod],[np.array(range(max_prod*max_sell))%max_sell]),0).transpose()
firm_params['cost']=lambda x: 0.5*(x**2)
firm_params['max_stock']=40
firm_params['verbose']=0
firm_params['explore_rate']=1
firm_params['explore_rate_decay']=0.9
firm_params['min_explore_rate']=0.05
firm_params['explore_turns']=40

if __name__ == '__main__':
    envir_glob=Environment(environment_params,firm_params)
    for simulation in range(N_SIMU):
        opt_prod_no_noise=[]
        for turn in range(N_TURN):
            envir_glob.act()
        opt_prod=envir_glob.price_memory[1:]
        envir_glob.plot()
        envir_glob.reset()

        opt_prod=np.array(opt_prod)
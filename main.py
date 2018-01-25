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
N_SIMU=40

###Environment parameters
environment_params={}
environment_params['price_evolution']={'average':2,'seasonal':5,'period':12,'random':2,'trend':0.05}
environment_params['n_firm']=1

###Firm parameters
firm_params={}
firm_params['WACC']=1
firm_params['initial_funds']=1000
firm_params['replay_memory_size']=1000
firm_params['memory_size']=20
firm_params['plot_frequency']=1
firm_params['possible_actions']=np.array(range(20)).reshape(-1,1)
firm_params['cost']=lambda x: 0.5*(x**2)
firm_params['max_stock']=10
firm_params['verbose']=0

if __name__ == '__main__':
    envir_glob=Environment(environment_params,firm_params)
    for simulation in range(N_SIMU):
        opt_prod_no_noise=[]
        for turn in range(N_TURN):
            envir_glob.act()
            opt_prod_no_noise+=[envir_glob.update_price(return_noise=False,save_price=False)]
        opt_prod=envir_glob.price_memory[1:]
        envir_glob.reset()
        print('OPTIMAL FUNDS KNOWING PRICE',1000+np.sum((opt_prod-0.5*np.floor(opt_prod))*np.floor(opt_prod)))
        print('OPTIMAL FUNDS NOT KNOWING PRICE',1000+np.sum((opt_prod-0.5*np.floor(opt_prod_no_noise))*np.floor(opt_prod_no_noise)))
        opt_prod=np.array(opt_prod)

import numpy as np
from Firm import Firm
import matplotlib.pyplot as plt
"""
This file contains the definition of the environment
in which the agents are run.
"""


class Environment:
    # List of the possible actions by the agents
    possible_actions = []

    def __init__(self,environment_params,firm_params):
        self.price_evolution=environment_params['price_evolution']
        self.current_period=0
        self.simulated_period=0
        self.firms={}
        self.price_memory=[0]
        self.inventory_evolution=[]
        self.sales_evolution=[]
        self.production_evolution=[]
        observation_size=self.observe().shape[0]
        for i in range(environment_params['n_firm']):
            self.add_firm('firm' + str(i),firm_params,observation_size)
    
    def add_firm(self,name,firm_params,observation_size):
        self.firms[name]=Firm(firm_params,observation_size)

    def reset(self):
        self.simulated_period+=1
        if self.simulated_period%10==0:
            self.plot_evolution()
        for firm in self.firms:
            current_firm=self.firms[firm]
            current_firm.reset()
        self.current_period=0
        self.price_memory=[0]
        self.price_memory=[0]
        self.inventory_evolution=[]
        self.sales_evolution=[]
        self.production_evolution=[]
        

    def observe(self):
        return np.array([(0.,)])
    
    def update_price(self,return_noise=True,save_price=True):
        price=self.price_evolution['average']+self.price_evolution['trend']*self.current_period
        price+=self.price_evolution['seasonal']*np.cos(2*3.14*self.current_period/self.price_evolution['period'])
        if return_noise:
            price+=np.random.normal(0,self.price_evolution['random'])
        if save_price:
            self.price_memory+=[max(price,0)]
        return max(price,0)
    
    def revenues(self,quantity):
        return self.price_memory[-1]*quantity
    
    def cost(self,quantity,firm):
        return firm.cost(quantity)
    
    def plot_evolution(self):
        plt.plot(self.inventory_evolution)
        plt.plot(self.price_memory[1:])
        plt.show()
        plt.plot(self.sales_evolution)
        plt.plot(self.price_memory[1:])
        plt.show()
        
        
    def act(self):
        self.current_period+=1
        self.update_price()
        for firm in self.firms:
            current_firm=self.firms[firm]
            action=current_firm.act(self.observe())
            purchase=action[0]
            production_order=action[1]
            current_firm.update_production_queue(production_order)
            current_firm.produce()
            sales=current_firm.get_sales(purchase)
            firm_reward=self.revenues(sales)-self.cost(production_order,current_firm)
            current_firm.funds+=firm_reward
            current_firm.reward(self.observe(),action,firm_reward)
            self.inventory_evolution+=[current_firm.inventory]
            self.sales_evolution+=[sales]
            self.production_evolution+=[production_order]
        
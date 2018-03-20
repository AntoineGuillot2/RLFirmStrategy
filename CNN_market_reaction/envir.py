
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
        self.inventory_evolution={}
        self.sales_evolution={}
        self.production_evolution={}
        self.funds_evolution={}
        observation_size=self.observe().shape[0]
        for i in range(environment_params['n_firm']):
            self.add_firm('firm' + str(i),firm_params,observation_size)
    
    def add_firm(self,name,firm_params,observation_size):
        self.firms[name]=Firm(firm_params,observation_size)
        self.inventory_evolution[name]=[]
        self.sales_evolution[name]=[]
        self.production_evolution[name]=[]
        self.funds_evolution[name]=[]

    def reset(self):
        self.simulated_period+=1
        if self.simulated_period%10==0:
            self.plot_evolution()
        for firm in self.firms:
            current_firm=self.firms[firm]
            current_firm.reset()
        self.current_period=0
        self.price_memory=[0]
        for firm in self.firms:
            self.inventory_evolution[firm]=[]
            self.sales_evolution[firm]=[]
            self.production_evolution[firm]=[]
            self.funds_evolution[firm]=[]

    def observe(self):
        return np.array([(self.price_memory[-1],)])
    
    def update_price(self,production=0,return_noise=True,save_price=True):
        price=self.price_evolution['average']+self.price_evolution['trend']*self.current_period
        price+=-np.abs(self.price_evolution['seasonal']*np.cos(2*3.14*self.current_period/self.price_evolution['period'])*production+1)
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
        for firm in self.firms:
            plt.plot(self.inventory_evolution[firm])
            plt.plot(self.price_memory[1:])
            plt.title(firm+' (stock)')
            plt.show()
            plt.plot(self.sales_evolution[firm])
            plt.title(firm+' (sales)')
            plt.plot(self.price_memory[1:])
            plt.show()
            plt.plot(self.funds_evolution[firm])
            plt.title(firm+' (funds)')
            plt.plot(self.price_memory[1:])
            plt.show()
        
        
    def act(self):
        self.current_period+=1
        production_order={}
        sales={}
        for firm in self.firms:
            current_firm=self.firms[firm]
            if current_firm.bankrupt==False:
                action=current_firm.act(self.observe())
                purchase=action[0]
                production_order[firm]=action[1]
                current_firm.update_production_queue(production_order[firm])
                current_firm.produce()
                sales[firm]=current_firm.get_sales(purchase)
        total_production=sum(sales.values())
        self.update_price(total_production)
        for firm in self.firms:
            current_firm=self.firms[firm]
            if current_firm.bankrupt==False:
                firm_reward=self.revenues(sales[firm])-self.cost(production_order[firm],current_firm)
                current_firm.funds+=firm_reward
                current_firm.reward(self.observe(),action,firm_reward)
                self.inventory_evolution[firm]+=[current_firm.inventory]
                self.funds_evolution[firm]+=[current_firm.funds]
                self.sales_evolution[firm]+=[sales[firm]]
                self.production_evolution[firm]+=[production_order[firm]]
        
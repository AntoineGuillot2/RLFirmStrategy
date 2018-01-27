
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
        self.firms={}
        self.price_memory=[0]
        self.production_hist=[]
        self.sales_hist=[]
        self.order_hist=[]
        self.stock_hist=[]
        self.purchase=[]
        self.order=[]
        self.reward_hist=[]
        observation_size=self.observe().shape[0]
        for i in range(environment_params['n_firm']):
            self.add_firm('firm' + str(i),firm_params,observation_size)
    
    def add_firm(self,name,firm_params,observation_size):
        self.firms[name]=Firm(firm_params,observation_size)

    def reset(self):
        for firm in self.firms:
            current_firm=self.firms[firm]
            current_firm.reset()
        self.current_period=0
        self.price_memory=[0]
        self.production_hist=[]
        self.sales_hist=[]
        self.order_hist=[]
        self.purchase=[]
        self.stock_hist=[]
        self.order=[]
        self.reward_hist=[]

    def observe(self):
        return np.array([(self.price_memory[-1]/10-0.5,)])
    
    def update_price(self,return_noise=True,save_price=True):
        price=self.price_evolution['average']
        price+=self.price_evolution['seasonal']*np.cos(3.14+2*3.14*self.current_period/self.price_evolution['period'])
        if return_noise:
            price+=np.random.normal(0,self.price_evolution['random'])
        if save_price:
            self.price_memory+=[max(price,0)]
        return max(price,0)
    
    def revenues(self,quantity):
        return self.price_memory[-1]*quantity
    
    def cost(self,quantity,firm):
        return firm.cost(quantity)
    
    def plot(self):
        plt.plot(self.purchase)
        plt.plot(self.price_memory[1:])
        plt.ylabel('Sales Evolution')
        plt.show()
        plt.plot(np.cumsum(self.reward_hist))
        plt.ylabel('Reward Evolution')
        plt.show()

        plt.plot(self.production_hist)
        plt.ylabel('Production Evolution')
        plt.show()
        plt.plot(self.stock_hist)
        plt.plot(self.price_memory[1:])
        plt.ylabel('Stock Evolution')
        plt.show()

        
        
        
    def act(self):
        self.current_period+=1
        self.update_price()
        for firm in self.firms:
            current_firm=self.firms[firm]
            action=current_firm.act(self.observe())
            order,sales=action
            self.purchase+=[sales]
            self.order+=[order]
            real_sales=current_firm.update_stock(current_firm.get_production(),sales)
            self.production_hist+=[current_firm.get_production()]
            self.sales_hist+=[real_sales]
            self.stock_hist+=[current_firm.stock]
            current_firm.update_production_planning(order)
            firm_reward=self.revenues(real_sales)-self.cost(order,current_firm)
            self.reward_hist+=[firm_reward]
            current_firm.funds+=firm_reward
            current_firm.reward(self.observe(),action,firm_reward)
        
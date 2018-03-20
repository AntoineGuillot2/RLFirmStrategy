# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:51:48 2017

@author: Antoi
"""

import numpy as np

def price(intercept,slope):
    return lambda x : intercept + slope *x

def cost(intercept,slope):
    return lambda x : intercept + slope *x**2

class environnement:
    
    def __init__(self):
        self.n_turn=0
        self.intercept=100
        self.slope=-1
        self.market_price=price(self.intercept,self.slope)
        
    def give_revenue(self,quantity,firm,discount):
        production_cost=firm.cost(quantity)
        revenue=self.market_price(quantity)*quantity
        firm.last_profit=revenue-production_cost
        firm.reward+=(1-discount)*revenue
        firm.money+=firm.last_profit
        firm.last_production=quantity
    
    def simulate(self,firm,quantity,discount):
        self.n_turn+=1
        self.intercept=100
        self.slope=-1
        self.market_price=price(self.intercept,self.slope)
        self.give_revenue(quantity,firm,discount)
        return firm.reward
        

class firm:
    
    def __init__(self):
        self.last_profit=20
        self.money=100
        self.cost=cost(0,1)
        self.last_production=10
        self.reward=0
        
    def get_state(self):
        return np.array([self.last_profit, self.money , self.last_production])
    
    def set_reward(self,reward):
        self.reward=reward
    


import tensorflow as tf
x = tf.placeholder('float',[1,3])
y = tf.placeholder('float',[None,15])
layer_hidden = tf.layers.dense(x,50,
    activation=tf.nn.relu,kernel_initializer=tf.random_normal_initializer)
layer_out = tf.layers.dense(layer_hidden,15,
    activation=tf.nn.relu,kernel_initializer=tf.random_normal_initializer)
selected_action=tf.argmax(layer_out,1)

loss= tf.reduce_mean(tf.square(layer_out - y))
trainer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)
init = tf.initialize_all_variables()

epsilon =1
discount = 0.05
discount_Q = 0.2
n_simulation = 100
with tf.Session() as sess:
    sess.run(init)
    for i in range(n_simulation):
        
        firm1=firm()
        envir1=environnement()
        current_state=firm1.get_state().reshape((1,3))
        current_state=current_state/max(current_state+1)
        j = 0
        while j < 200:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            quantity,allQ = sess.run([selected_action,layer_out],feed_dict={x:current_state})
            quantity=quantity[0]
            
            if np.random.rand(1) < epsilon:
                quantity= min(np.int(np.random.exponential(10)),14)
            #Get new state and reward from environment
            reward=envir1.simulate(firm1,quantity,discount)
            #Obtain the Q' values by feeding the new state through our network
            new_state= firm1.get_state().reshape((1,3))
            new_state=new_state/max(new_state+1)
            Q1 = sess.run(layer_out,feed_dict={x:new_state})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ/max(allQ)
            targetQ[0,quantity] = (1-discount_Q)*reward + discount_Q*maxQ1
            #Train our network using target and predicted Q values
            sess.run(updateModel,feed_dict={x:current_state,y :targetQ})
            current_state = new_state
        epsilon=epsilon/2
        print(firm1.money)
        print(firm1.reward)
        

    


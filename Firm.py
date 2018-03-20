#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:47:13 2018

@author: antoine
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input,Dense


class Firm:
    def __init__(self,params,observation_size=12):
        self.initial_funds=params['initial_funds']
        self.funds=self.initial_funds
        self.WACC=params['WACC']
        
        self.plot_frequency=params['plot_frequency']
        self.possible_actions=params['possible_actions']
        self.env_obs_size=observation_size
        self.memory_size=params['memory_size']
        self.init_memory()
        self.init_Q_estimator(observation_size+self.state_observation().shape[1])
        
        self.replay_memory_size=params['replay_memory_size']
        self.init_replay_memory()


        
        self.played_games=0
        self.list_rewards=[]
        self.funds_evolution=[]
        self.list_actions=[]

        
    def state_observation(self):
        return self.event_memory.reshape(1,-1)
        
        
    def init_Q_estimator(self,n_input):
        self.Q_estimator_shape=(n_input+self.possible_actions.shape[1],1)
        input_data = Input(shape=(n_input+self.possible_actions.shape[1],))
        x = Dense(20,activation='relu')(input_data)
        x = Dense(20,activation='relu')(x)
        estimated_Q = Dense(1,activation='linear')(x)
        self.Q_estimator = Model(inputs=input_data, outputs=estimated_Q)
        self.Q_estimator.compile(optimizer='rmsprop',
                      loss='mse')

        
    def init_replay_memory(self):
        self.replay_memory=np.zeros((0,self.Q_estimator_shape[0]+1))
        
    def update_replay_memory(self,new_observation):
        self.replay_memory=np.concatenate((self.replay_memory,new_observation),0)
        if np.shape(self.replay_memory)[0]>self.replay_memory_size:
            self.replay_memory=self.replay_memory[1:]
        
    def init_memory(self):
        self.event_memory=np.zeros((1,self.memory_size))
        
    def update_memory(self,new_observation):
        new_observation_size=new_observation.shape[0]
        self.event_memory=np.concatenate((new_observation,self.event_memory.reshape(1,-1)),1)[0][:self.memory_size*new_observation_size]

    def cost(self,quantity):
        return 0.5*(quantity**2)
        
    def reset(self):
        print("GAME NUMBER: ", self.played_games)
        print('*****FINAL FUNDS******')
        print(self.funds)
        print(self.replay_memory)
        self.Q_estimator.fit(self.replay_memory[:,:-1],self.replay_memory[:,-1],epochs=100+100*self.played_games,initial_epoch=100*self.played_games,verbose=0)
        self.played_games+=1
        if (self.played_games%self.plot_frequency)==0:
            plt.plot(self.list_actions)
            plt.title("Production Evolution")
            
        self.list_rewards=[]
        self.funds_evolution=[]
        self.list_actions=[]
        self.funds=self.initial_funds
        self.init_memory()
        pass
    
    def compute_best_action2(self,observation):
        values=[]
        for action in self.possible_actions:
            state_action=np.concatenate((observation.reshape(-1,1),action.reshape(-1,1)),0).reshape(1,-1)
            values+=[self.Q_estimator.predict(state_action)[0][0]]
        return self.possible_actions[np.argmax(values)], np.max(values)
    
    def compute_best_action(self,observation):
        observation2=np.repeat(observation.reshape(-1,1),len(self.possible_actions),0)
        values=[]
        values=self.Q_estimator.predict(self.possible_actions.reshape(-1,1),observation2)
        print(values)
        return self.possible_actions[np.argmax(values)], np.max(values)

    def act(self, observation):
        observation=observation.reshape((1,-1))
        observation=np.concatenate((observation,self.state_observation()),1)
        self.previous_observation=observation
        if (self.played_games<30):
            if np.random.uniform()<0.05:
                random_action=np.random.randint(len(self.possible_actions))
                self.list_actions+=[self.possible_actions[random_action]]
            else:
                self.list_actions+=[self.compute_best_action(observation)[0]]
        else:
            self.list_actions+=[self.compute_best_action(observation)[0]]
        return self.list_actions[-1]
            
    def reward(self, observation, action, reward):
        self.list_rewards+=[reward]
        self.funds_evolution+=[self.funds]
        observation=observation.reshape((1,-1))
        observation=np.concatenate((observation,self.state_observation()),1)
        target=reward+(1-self.WACC)*self.compute_best_action(observation)[1]
        new_observation=np.concatenate((self.previous_observation.reshape(1,-1),action.reshape(1,-1),np.array([[target]])),1)
        new_observation=new_observation.reshape((-1,self.Q_estimator_shape[0]+1))
        self.update_replay_memory(new_observation)
        self.update_memory(observation[:,:self.env_obs_size])
        pass
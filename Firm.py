#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:47:13 2018

@author: antoine
"""
import numpy as np
from keras.models import Model
from keras.layers import Input,Dense, Conv1D, MaxPooling1D, Concatenate, Flatten


class Firm:
    def __init__(self,params,observation_size=12):
        print('NEW INIT')
        self.initial_funds=params['initial_funds']
        self.last_action=0
        self.funds=self.initial_funds
        self.stock=0
        self.current_reward=0.
        self.WACC=params['WACC']
        self.cost=params['cost']
        
        self.plot_frequency=params['plot_frequency']
        self.possible_actions=params['possible_actions']
        self.env_obs_size=observation_size
        self.memory_size=params['memory_size']
        self.init_event_memory()
        self.init_Q_estimator(observation_size)
        
        self.replay_memory_size=params['replay_memory_size']
        self.init_replay_memory()

        
        self.played_games=0
        self.list_rewards=[]
        self.funds_evolution=[]
        self.list_actions=[]

    def get_state(self):
        return np.array([(self.funds/2000,self.stock/10,self.current_reward,self.last_action)])
        
        
    def init_Q_estimator(self,n_input):
        self.Q_estimator_shape=(n_input,self.possible_actions.shape[1],1)
        input_actions=Input(shape=(self.possible_actions.shape[1],))
        x_actions = Dense(10,activation='relu')(input_actions)
        input_data = Input(shape=(self.memory_size,n_input+self.get_state().shape[1],))
        x_data = Conv1D(10,3)(input_data)
        x_data = MaxPooling1D(3)(x_data)
        x_data=Flatten()(x_data)
        x=Concatenate()([x_data,x_actions])
        x=Dense(10,activation='relu')(x)
        estimated_Q = Dense(1,activation='linear')(x)
        self.Q_estimator = Model(inputs=[input_actions,input_data], outputs=estimated_Q)
        self.Q_estimator.compile(optimizer='nadam',
                      loss='mse')

    ###Initialize the replay memory
    def init_replay_memory(self):
        self.action_memory=np.zeros((0,self.Q_estimator_shape[1]))
        self.observation_memory=np.zeros((0,self.event_memory.shape[1],self.event_memory.shape[2]))
        self.Q_memory=np.zeros((0,self.Q_estimator_shape[2]))
        
    def update_replay_memory(self,action,state_observation,Q_value):
        self.observation_memory=np.concatenate((self.observation_memory,state_observation),0)
        self.action_memory=np.concatenate((self.action_memory,action),0)
        Q_value=np.array([(Q_value,)])
        self.Q_memory=np.concatenate((self.Q_memory,Q_value),0)
        if np.shape(self.action_memory)[0]>self.replay_memory_size:
            self.observation_memory=self.observation_memory[1:]
            self.action_memory=self.action_memory[1:]
            self.Q_memory=self.Q_memory[1:]
        
    def init_event_memory(self):
        self.event_memory=np.zeros((1,self.memory_size,self.env_obs_size+self.get_state().shape[1]))
        
    def update_event_memory(self,new_observation):
        new_event_memory=np.concatenate((new_observation.reshape((1,1,-1)),self.event_memory),1)
        self.event_memory=new_event_memory[:,:self.memory_size,:]
        
    def reset(self):
        self.Q_estimator.fit([self.action_memory,self.observation_memory],self.Q_memory,epochs=200,verbose=0)
        self.played_games+=1
        
        print(self.played_games)
        print(self.funds)
        self.rewards=0
        self.funds=self.initial_funds
        self.init_event_memory()
    
    def compute_best_action(self,observation):
        values=[]
        for action in self.possible_actions:
            values+=[self.Q_estimator.predict([action,observation])[0][0]]
        return self.possible_actions[np.argmax(values)], np.max(values)

    def act(self, observation):
        state_observation=np.concatenate((self.get_state(),observation),1)
        self.previous_observation=state_observation
        self.update_event_memory(state_observation)
        if (self.played_games<30):
            if np.random.uniform()<0.05:
                random_action=np.random.randint(len(self.possible_actions))
                action = self.possible_actions[random_action]
            else:
                action = self.compute_best_action(self.event_memory)[0]
        else:
            action = self.compute_best_action(self.event_memory)[0]
        self.last_action=action
        return action
            
    def reward(self, observation, action, reward):
        self.current_reward=reward/5
        target=reward+(1-self.WACC)*self.compute_best_action(self.event_memory)[1]
        self.update_replay_memory(action.reshape(1,-1),self.event_memory,target)

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
        
        
        
        ###Initial firm parameters
        self.initial_funds=params['initial_funds']
        self.funds=self.initial_funds
        self.initial_inventory=params['initial_inventory']
        self.inventory=params['initial_inventory']
        self.max_inventory=params['max_inventory']
        self.production_queue=[0]*params['production_time']
        self.current_reward=0.
        self.WACC=params['WACC']
        self.cost=params['cost']
        self.production_time=params['production_time']
        self.epsilon_greedy=params['epsilon_greedy']
        ##Set explore rate evolution
        self.explore_rate=params['initial_explore_rate']
        self.explore_rate_decay=params['explore_rate_decay']
        self.min_explore_rate=params['min_explore_rate']
        self.explore_turns=params['explore_turns']
        self.temperature=10
        
        self.bankrupt=False
        
        ##Scaling parameters:
        self.mean_action=0
        self.std_action=1
        self.mean_obs=0
        self.std_obs=1
        self.mean_Q=0
        self.std_Q=1
        
        
        self.plot_frequency=params['plot_frequency']
        self.possible_actions=params['possible_actions']
        self.last_action=self.possible_actions[0]
        self.env_obs_size=observation_size
        self.memory_size=params['memory_size']
        self.init_event_memory()
        self.init_Q_estimator(observation_size)
        
        self.epochs=params['epochs']
        
        self.replay_memory_size=params['replay_memory_size']
        self.init_replay_memory()

        self.n_step=0
        self.played_games=0
        
        
        self.list_rewards=[]
        self.funds_evolution=[]
        self.list_actions=[]

    def get_state(self):
        return np.array([(self.funds,self.current_reward,self.inventory,*self.last_action)])
        
        
    def init_Q_estimator(self,n_input):
        self.Q_estimator_shape=(n_input,self.possible_actions.shape[1],1)
        self.input_actions=Input(shape=(self.possible_actions.shape[1],))
        self.x_actions = Dense(10,activation='relu')(self.input_actions)
        self.input_data = Input(shape=(self.memory_size,n_input+self.get_state().shape[1],))
        self.x_data = Conv1D(10,3,activation='relu')(self.input_data)
        self.x_data = Conv1D(10,3,activation='relu')(self.x_data)
        self.x_data = MaxPooling1D(3)(self.x_data)
        self.x_data=Flatten()(self.x_data)
        self.x=Concatenate()([self.x_data,self.x_actions])
        self.x=Dense(10,activation='relu')(self.x)
        self.estimated_Q = Dense(1,activation='linear')(self.x)
        self.Q_estimator = Model(inputs=[self.input_actions,self.input_data], outputs=self.estimated_Q)
        self.Q_estimator.compile(optimizer='rmsprop',
                      loss='mse')
        
    def estimate_Q(self,actions,observations,rescale=True):
        actions=(actions-self.mean_action)/self.std_action
        observations=(observations-self.mean_obs)/self.std_obs
        return self.Q_estimator.predict([actions,observations])[:,0]
    
    def fit_Q(self,rescale=True):
        if rescale:
            action_memory=(self.action_memory-self.mean_action)/self.std_action
            observation_memory=(self.observation_memory-self.mean_obs)/self.std_obs
            Q_memory=(self.Q_memory-self.mean_Q)/self.std_Q
            self.Q_estimator.fit([action_memory,observation_memory],Q_memory,epochs=self.epochs,verbose=0)
        
        

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
            
    def update_scaling(self):
        self.mean_obs=0.5*self.mean_obs+0.5*np.mean(self.observation_memory,0)
        self.std_obs=0.5*self.std_obs+0.5*np.std(self.observation_memory,0)
        
        self.mean_action=0.5*self.mean_action+0.5*np.mean(self.action_memory,0)
        self.std_action=0.5*self.std_action+0.5*np.std(self.action_memory,0)
        
        self.std_Q=0.5*self.std_Q+np.std(self.Q_memory,0)*0.5
        self.mean_Q=0.5*self.mean_Q+np.mean(self.Q_memory,0)*0.5
        
    def init_event_memory(self):
        self.event_memory=np.zeros((1,self.memory_size,self.env_obs_size+self.get_state().shape[1]))
        
    def update_event_memory(self,new_observation):
        new_event_memory=np.concatenate((new_observation.reshape((1,1,-1)),self.event_memory),1)
        self.event_memory=new_event_memory[:,:self.memory_size,:]
        
    def reset(self):
        self.bankrupt=False
        self.update_scaling()
        if self.explore_rate>self.min_explore_rate:
            self.explore_rate*=self.explore_rate_decay
        if self.temperature>0.1:
            self.temperature*=self.explore_rate_decay
        self.fit_Q()
        
        self.played_games+=1
        self.n_step=0
        print(self.played_games)
        print(self.funds)
        self.rewards=0
        self.inventory=self.initial_inventory
        self.funds=self.initial_funds
        self.init_event_memory()
    
    def compute_best_action(self,observation):
        observation=np.repeat(observation,len(self.possible_actions),0)
        values=self.estimate_Q(self.possible_actions,observation)
        return self.possible_actions[np.argmax(values)], np.max(values)
    
    def compute_action_values(self,observation):
        observation=np.repeat(observation,len(self.possible_actions),0)
        values=self.estimate_Q(self.possible_actions,observation)
        return values

    def act(self, observation):
        if self.funds<0:
            self.bankrupt=True
            print('Firm bankruptcy')
        self.n_step+=1
        state_observation=np.concatenate((self.get_state(),observation),1)
        self.previous_observation=state_observation
        self.update_event_memory(state_observation)
        if self.epsilon_greedy:
            if (self.played_games<self.explore_turns):
                if np.random.uniform()<self.explore_rate:
                    random_action=np.random.randint(len(self.possible_actions))
                    action = self.possible_actions[random_action]
                else:
                    action = self.compute_best_action(self.event_memory)[0]
            else:
                action = self.compute_best_action(self.event_memory)[0]
        else:
            values=np.exp(self.compute_action_values(self.event_memory)/self.temperature)
            values=values/np.sum(values)
            action = self.possible_actions[np.random.choice(values.shape[0],p=values)]
            
        self.last_action=action
        return action
            
    def reward(self, observation, action, reward):
        self.current_reward=reward
        if self.funds<0:
            target=-1000

        else:
            target=reward+(1-self.WACC)*self.compute_best_action(self.event_memory)[1]
        self.update_replay_memory(action.reshape(1,-1),self.event_memory,target)
        
    def update_production_queue(self,production):
        self.production_queue=np.concatenate((self.production_queue,[production]))
        
    def produce(self):
        self.inventory+=self.production_queue[0]
        if self.inventory>=self.max_inventory:
            self.inventory=self.max_inventory
        self.production_queue=self.production_queue[1:]
    
    def get_sales(self,purchase):
        if self.inventory-purchase>=0:
            self.inventory=self.inventory-purchase
            return purchase
        else:
            self.inventory=0
            return self.inventory
        
    def save_model(self,file):
        self.Q_estimator.save_weights(file)
    
    def load_model(self,file):
        self.Q_estimator.load_weights(file)

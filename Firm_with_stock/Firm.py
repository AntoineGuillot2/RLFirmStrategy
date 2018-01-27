#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:47:13 2018

@author: antoine
"""
import numpy as np
from keras.models import Model
from keras.layers import Input,Dense, Conv1D, MaxPooling1D, Concatenate, Flatten, BatchNormalization, Dropout


class Firm:
    def __init__(self,params,observation_size=12):
        print('NEW INIT')
        self.initial_funds=params['initial_funds']
        self.last_action=(0,0)
        self.funds=self.initial_funds
        self.stock=20
        self.max_stock=params['max_stock']
        self.production_planning=[0]*4
        self.current_reward=0.
        self.WACC=params['WACC']
        self.cost=params['cost']
        self.explore_rate=params['explore_rate']
        self.explore_rate_decay=params['explore_rate_decay']
        self.explore_turns=params['explore_turns']
        self.min_explore_rate=params['explore_turns']
        
        self.plot_frequency=params['plot_frequency']
        self.possible_actions=params['possible_actions']
        self.env_obs_size=observation_size
        self.memory_size=params['memory_size']
        self.init_event_memory()
        self.init_Q_estimator(observation_size)
        self.init_Q_learner(observation_size)
        self.replay_memory_size=params['replay_memory_size']
        self.init_replay_memory()
        self.mean_action=0
        self.mean_obs=0
        self.std_action=1
        self.std_obs=1
        self.n_step=0

        
        self.played_games=0
        self.list_rewards=[]
        self.funds_evolution=[]
        self.list_actions=[]

    def get_state(self):
        return np.array([(self.funds,self.stock,self.current_reward)])
        
        
    def init_Q_estimator(self,n_input):
        self.Q_estimator_shape=(n_input,self.possible_actions.shape[1],1)
        input_actions=Input(shape=(self.possible_actions.shape[1],))
        input_data = Input(shape=(self.memory_size,n_input+self.get_state().shape[1],))
        x_data = Conv1D(10,12)(input_data)
        x_data = BatchNormalization()(x_data)
        x_data = Conv1D(20,6)(input_data)
        x_data = BatchNormalization()(x_data)
        x_data= Flatten()(x_data)
        x=Concatenate()([x_data,input_actions])
        x=Dense(30,activation='relu')(x)
        x=Dense(30,activation='relu')(x)
        estimated_Q = Dense(1,activation='linear')(x)
        self.Q_estimator = Model(inputs=[input_actions,input_data], outputs=estimated_Q)
        self.Q_estimator.compile(optimizer='nadam',
                      loss='mse')
        
    def init_Q_learner(self,n_input):
        input_actions=Input(shape=(self.possible_actions.shape[1],))
        input_data = Input(shape=(self.memory_size,n_input+self.get_state().shape[1],))
        x_data = Conv1D(10,12)(input_data)
        x_data = BatchNormalization()(x_data)
        x_data = Conv1D(20,6)(input_data)
        x_data = BatchNormalization()(x_data)
        x_data= Flatten()(x_data)
        x=Concatenate()([x_data,input_actions])
        x=Dense(30,activation='relu')(x)
        x=Dense(30,activation='relu')(x)
        estimated_Q = Dense(1,activation='linear')(x)
        self.Q_learner = Model(inputs=[input_actions,input_data], outputs=estimated_Q)
        self.Q_learner.compile(optimizer='nadam',
                      loss='mse')
    
    def drift_Q_estimator(self,lr=0.1):
        Q_learner_weights=self.Q_learner.get_weights()
        Q_estimator_weights=self.Q_estimator.get_weights()
        self.Q_estimator.set_weights(np.array([(1-lr)*Q_estimator_weights[0]+lr*Q_learner_weights[0]]))
        
        
    def fit_Q_learner(self,observation_memory,action_memory,Q_memory,batch_prop=0.5,n_epoch=1):
        if len(self.action_memory)<self.replay_memory_size:
            batch_prop=min(1,(self.replay_memory_size*batch_prop)/self.action_memory.shape[0])
        

        self.mean_action=0.9*self.mean_action+0.1*np.mean(action_memory,0)
        self.mean_obs=0.9*self.mean_obs+0.1*np.mean(observation_memory,0)
        self.std_action=0.9*self.std_action+0.1*np.std(action_memory,0)
        self.std_obs=0.9*self.std_obs+0.1*np.std(observation_memory,0)

        batch_index=np.random.randint(0,high=len(observation_memory),size=int(len(observation_memory)*batch_prop))
        observation_batch=(observation_memory[batch_index]-self.mean_obs)/self.std_obs
        action_batch=(action_memory[batch_index]-self.mean_action)/self.std_action
        Q_batch=Q_memory[batch_index]
        self.Q_learner.fit([action_batch,observation_batch],Q_batch,epochs=n_epoch,verbose=1)
        
    def estimate_Q(self,action,observation):
        if self.played_games>0:
            observation=(observation-self.mean_obs)/self.std_obs
            action=(action-self.mean_action)/self.std_action
            return self.Q_estimator.predict([action,observation])
        else:
            return np.zeros((np.shape(action)[0],1))
      

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
        self.n_step=0
        self.fit_Q_learner(self.observation_memory,self.action_memory,self.Q_memory)
        self.drift_Q_estimator(0.15)
        self.played_games+=1
        
        print(self.played_games)
        print(self.funds)
        self.rewards=0
        self.funds=self.initial_funds
        self.init_event_memory()
        
    
    def compute_best_action2(self,observation):
        observation2=np.repeat(observation,len(self.possible_actions),0)
        values=[]
        if self.n_step==99:
            print("**************VECTOR*********************")
            print(observation2[5])
            print("**************LOOP*********************")
            print(observation)
        values=self.estimate_Q(self.possible_actions,observation2)[:,0]
        return self.possible_actions[np.argmax(values)], np.max(values)
    
    def compute_best_action(self,observation):
        values=[]
        for action in self.possible_actions:
            values+=[self.estimate_Q([action],observation)]
        return self.possible_actions[np.argmax(values)], np.max(values)
    
    def compute_action_values(self,observation):
        observation=np.repeat(observation,len(self.possible_actions),0)
        values=self.estimate_Q(self.possible_actions,observation)
        return self.possible_actions, values
    
    def update_stock(self,production,sales):
        stock_evolution=production-sales
        if self.stock+stock_evolution<0:
            sales=self.stock+production
            self.stock=0
            real_sales=sales
        elif self.stock+stock_evolution>self.max_stock:
            self.stock=self.max_stock
            real_sales=sales
        else:
            self.stock+=stock_evolution
            real_sales=sales
        return(real_sales)
        
    def update_production_planning(self,production):
        self.production_planning=np.concatenate((self.production_planning[1:],np.array([production])))
    
    def get_production(self):
        return self.production_planning[0]

    def act(self, observation):
        self.n_step+=1
        state_observation=np.concatenate((self.get_state(),observation),1)
        self.previous_observation=state_observation
        if self.played_games<self.min_explore_rate:
            if np.random.uniform()<=self.explore_rate:
                action=self.possible_actions[np.random.choice(len(self.possible_actions))]
                
            else:
                action=self.compute_best_action(self.event_memory)[0]
        else:
            action=self.compute_best_action(self.event_memory)[0]
        self.last_action=action.reshape(1,-1)
        return action     
            
    def reward(self, observation, action, reward):
        self.current_reward=reward
        state_observation=np.concatenate((self.get_state(),observation),1)
        self.update_event_memory(state_observation)
        target=reward+(1-self.WACC)*self.compute_best_action(self.event_memory)[1]
        self.update_replay_memory(action.reshape(1,-1),self.event_memory,target)

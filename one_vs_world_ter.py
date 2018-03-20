# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:32:11 2017

@author: Antoi
"""
import numpy as np
import numpy.random as rd
class environnement:
    
    def __init__(self,intercept=50,slope=-1,moving_intercept=25):
        self.firms=[]
        self.period=0
        self.currentMarketPrice=self.marketPrice(intercept,slope,moving_intercept)
    
    def marketPrice(self,intercept,slope,moving_intercept=0):
        return lambda x: max(0,intercept+moving_intercept*np.sin(3.14*self.period/12) + slope*12)
    
    def addFirm(self,name,intercept=10,slope=2,power=2,wacc=0.05,memory=1):
        new_firm=firm(name,intercept,slope,power,wacc,memory)
        self.firms=self.firms+[new_firm]
        
    def giveRevenue(self,firm,sold_quantity):
        firm.lastRevenue=sold_quantity*self.currentMarketPrice(sold_quantity)-firm.lastCost
        firm.money+=firm.lastRevenue
        if firm.money>0:
            firm.reward+=(1-firm.wacc)**self.period*firm.lastRevenue
        else:
            firm.reward+=-(1-firm.wacc)**self.period*500000
        
    def simulate(self,action):
        firm=self.firms[0]
        selectedQuantity=np.argmax(action)
        producedQuantity=firm.produce(selectedQuantity)
        firm.played_turn+=1
        self.giveRevenue(firm,producedQuantity)
        firm.saveState()
        self.period+=1
        
class firm:
    
    def __init__(self,name,intercept=30,slope=2,power=2,wacc=0.05,memory=1):
        self.name=name
        self.played_turn=0
        self.money=100000
        self.reward=0
        self.lastProduction=0
        self.lastRevenue=0
        self.lastCost=0
        self.wacc=wacc
        self.savedStates=[]
        self.memory=memory
        self.cost=self.prodCost(intercept,slope,power)
        
    def set_possibleAction(self,possibleAction):
        self.possibleAction=possibleAction
    
    def prodCost(self,intercept,slope,power):
        return lambda x: intercept + slope*x**power
    
    def produce(self,quantity):
        self.lastCost=self.cost(quantity)
        self.money+=(-self.lastCost)
        self.lastProduction=quantity
        return(quantity)
        
    def getState(self):
        zeroList=[0]*(18*self.memory)
        return np.array((zeroList+self.savedStates)[-(18*self.memory):]).reshape((1,18*self.memory))
        
    def saveState(self):
        time_period=[0]*12
        time_period[self.played_turn%12]=1
        self.savedStates=self.savedStates+time_period+[self.money,self.reward,self.lastCost,self.lastProduction,self.lastRevenue,self.played_turn%12]
        if len(self.savedStates)>18*self.memory:
            self.savedStates=self.savedStates[-(18*self.memory):]
        
    def actionSelection(self,actionValueEstimation):
        best_estimate=-10**22
        for action in self.possibleAction:
            actionState=np.concatenate((action.reshape(1,-1),self.getState().reshape(1,-1)),1)
            current_value=actionValueEstimation(actionState)[0]
            if best_estimate<current_value:
                best_estimate=current_value
                best_action=action
        return best_estimate[0], best_action

memory=3
envir=environnement(50,-4,25)
envir.addFirm("A",memory=memory)
currentState=envir.firms[0].getState()
possibleActions=np.identity(30)
envir.firms[0].set_possibleAction(possibleActions)
input_variables=currentState.shape[1]+envir.firms[0].possibleAction.shape[1]

from keras.models import Model
from keras.layers import Input,Dense, BatchNormalization, Dropout
from keras.initializers import he_normal, he_uniform
from keras.regularizers import l2

input_data = Input(shape=(input_variables,))
x=BatchNormalization()(input_data)
x = Dense(50,activation='relu')(x)
x = Dense(50,activation='relu')(x)
estimated_Q = Dense(1,activation='linear')(x)
Q_estimator = Model(inputs=input_data, outputs=estimated_Q)
Q_estimator.compile(optimizer='rmsprop',
              loss='mse')
   
bestQ,bestAction=envir.firms[0].actionSelection(Q_estimator.predict)

  
###Variante Monte-Carlo
import matplotlib.pyplot as plt
n_iterations=10
n_time_steps=200
production=[]
epsilon=0.05

for i in range(n_iterations):
    targetQ_save=None
    state_save=None
    envir=environnement(50,-4,25)
    envir.addFirm("A",wacc=0.05,memory=memory)
    envir.firms[0].set_possibleAction(possibleActions)
    production=[]
    starting_epoch=0
    for time_step in range(n_time_steps):
        print(time_step)
        
        ###Computing next best ation
        estimatedQ, nextAction=envir.firms[0].actionSelection(Q_estimator.predict)
        ###Exploration
        if rd.uniform(0,1)<epsilon:
            nextAction=nextAction*0
            nextAction[int(rd.uniform(0,29.99))]=1
        ##Saving state action
        stateAction=np.concatenate((nextAction.reshape(1,-1),envir.firms[0].getState().reshape(1,-1)),1)
        if (state_save is None):
            state_save=stateAction
        else:
            state_save=np.concatenate((state_save,stateAction),0)
        ##Save next action to plot
        production+=[np.argmax(nextAction)]
        
        ##Simulation new turn
        envir.simulate(nextAction)
        newState=envir.firms[0].getState()
        newReward=newState[0][-2]
        targetQ=estimatedQ
        if time_step<n_time_steps-1:
            nextQ, _ =envir.firms[0].actionSelection(Q_estimator.predict)
            targetQ=newReward+0.95*(nextQ)
        else:
            targetQ=newReward
        if targetQ_save is None:
            targetQ_save=np.array(estimatedQ).reshape(1,-1)
        else:
            targetQ_save=np.concatenate((targetQ_save,np.array(estimatedQ).reshape(1,-1)),0)
        if time_step>10:
            history=Q_estimator.fit(state_save,targetQ_save,verbose=1,epochs=100+starting_epoch,initial_epoch=starting_epoch)

    plt.plot(production)
    plt.show()
    starting_epoch+=10
    epsilon=epsilon*1/2
    
    

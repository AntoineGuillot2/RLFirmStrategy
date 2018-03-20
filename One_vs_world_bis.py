# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 09:26:53 2017

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
    
    def addFirm(self,name,intercept=10,slope=1,power=2,wacc=0.05,memory=1):
        new_firm=firm(name,intercept,slope,power,wacc,memory)
        self.firms=self.firms+[new_firm]
        
    def giveRevenue(self,firm,sold_quantity):
        firm.lastRevenue=sold_quantity*self.currentMarketPrice(sold_quantity)-firm.lastCost
        firm.money+=firm.lastRevenue
        if firm.money>0:
            firm.reward+=(1-firm.wacc)**self.period*firm.lastRevenue
        else:
            firm.reward+=-(1-firm.wacc)**self.period*500000
        
    def simulate(self,optimalMapper):
        for firm in self.firms:
            selectedQuantity=firm.actionSelection(optimalMapper)
            producedQuantity=firm.produce(selectedQuantity)
            firm.played_turn+=1
            self.giveRevenue(firm,producedQuantity)
            firm.saveState()
        self.period+=1
    
    

class firm:
    
    def __init__(self,name,intercept=30,slope=2,power=1,wacc=0.05,memory=1):
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
    
    def prodCost(self,intercept,slope,power):
        return lambda x: intercept + slope*x**power
    
    def produce(self,quantity):
        self.lastCost=self.cost(quantity)
        self.money+=(-self.lastCost)
        self.lastProduction=quantity
        return(quantity)
        
    def getState(self):
        zeroList=[0]*(17*self.memory)
        return np.array((zeroList+self.savedStates)[-(17*self.memory):]).reshape((1,17*self.memory))
        
    def saveState(self):
        time_period=[0]*12
        time_period[self.played_turn%12]=1
        self.savedStates=self.savedStates+time_period+[self.money,self.reward,self.lastCost,self.lastProduction,self.lastRevenue]
        if len(self.savedStates)>17*self.memory:
            self.savedStates=self.savedStates[-(17*self.memory):]
        
    def actionSelection(self,optimalMapper):
        return optimalMapper(self.getState)

n_variable=17
memory=1
envir=environnement(50,-4,25)
envir.addFirm("A",wacc=0.05,memory=memory,)
history=envir.firms[0].getState()
for i in range(100):
    selectedAction=5
    def actionMapper(states):
        return(selectedAction)
    envir.simulate(actionMapper)
    history=np.concatenate((history,envir.firms[0].getState()),0)
    
import matplotlib.pyplot as plt
plt.plot(history[:,-3])

from keras.models import Model
from keras.layers import Input,Dense, BatchNormalization, Dropout
from keras.initializers import he_normal, he_uniform
from keras.regularizers import l2

input_data = Input(shape=(n_variable*memory,))
x=BatchNormalization()(input_data)
x = Dense(30,activation='relu')(x)
estimated_Q = Dense(30,activation='relu')(x)
Q_estimator = Model(inputs=input_data, outputs=estimated_Q)
Q_estimator.compile(optimizer='rmsprop',
              loss='mse')
    
    
###Variante Monte-Carlo
import random as rd
n_iterations=10
n_time_steps=100
production=[]
epsilon=0
for i in range(n_iterations):
    targetQ_save=None
    state_save=None
    envir=environnement(50,-4,25)
    envir.addFirm("A",wacc=0.05,memory=memory)
    currentState=envir.firms[0].getState()
    production=[]

    for time_step in range(n_time_steps):
        print(time_step)
        estimatedQ=Q_estimator.predict(currentState)[0]
        if (state_save==None):
            state_save=currentState
            Q_save=estimatedQ
        else:
            state_save=np.concatenate((state_save,currentState),0)
            Q_save=np.concatenate((Q_save,estimatedQ),0)
        nextAction=np.argmax(estimatedQ)
        if rd.uniform(0,1)<epsilon:
            nextAction=int(rd.uniform(0,29.99))
        production+=[nextAction]
        envir.simulate(lambda x: nextAction)
        newState=envir.firms[0].getState()
        newReward=newState[0][-1]
        if time_step<n_time_steps-1:
            estimatedQ[nextAction]=newReward+0.95*(np.max(Q_estimator.predict(newState)[0]))
        else:
            estimatedQ[nextAction]=newState[0][-5]
        if targetQ_save==None:
            targetQ_save=np.array(estimatedQ).reshape(1,-1)
        else:
            targetQ_save=np.concatenate((targetQ_save,np.array(estimatedQ).reshape(1,-1)),0)
        currentState=newState
        if time_step>100:
            history=Q_estimator.fit(state_save,targetQ_save,verbose=0,epochs=20)
    plt.plot(production)
    plt.show()
    epsilon=epsilon/2


            
        

        

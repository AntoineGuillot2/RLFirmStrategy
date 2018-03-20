# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 21:42:53 2017

@author: Antoi
"""
import numpy as np 
class market:
    
    def __init__(self,market_init):
        self.period=0
        self.firms=[]
        self.marketPrice=self.set_marketPrice(market_init)
    
    def set_marketPrice(self,marketProperty):
        intercept=marketProperty["intercept"]
        slope=marketProperty["slope"]
        moving_intercept=marketProperty["moving_intercept"]
        return lambda x: max(0,intercept+moving_intercept*np.sin(3.14*self.period/12) + slope*12)
    
    def buyGoods(self,firm,quantity):
        revenues=self.marketPrice(quantity)*quantity
        firm.getMoney(revenues)
        
    def addFirm(self):
        


class firm:
    
    def __init__(self,name=,cost,memory=1):
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
        zeroList=[0]*(6*self.memory)
        return np.array((zeroList+self.savedStates)[-(6*self.memory):]).reshape((1,6*self.memory))
        
    def saveState(self):
        self.savedStates=self.savedStates+[self.money,self.reward,self.lastCost,self.lastProduction,self.lastRevenue,self.played_turn%12]
        if len(self.savedStates)>6*self.memory:
            self.savedStates=self.savedStates[-(6*self.memory):]
        
    def actionSelection(self,actionValueEstimation):
        best_estimate=-10**22
        for action in self.possibleAction:
            actionState=np.concatenate((action.reshape(1,-1),self.getState().reshape(1,-1)),1)
            current_value=actionValueEstimation(actionState)[0]
            if best_estimate<current_value:
                best_estimate=current_value
                best_action=action
        return best_estimate[0], best_action

    

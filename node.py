import matplotlib.pyplot as plt
from tqdm import trange
import os
import numpy as np
import math
from scipy import special
import skimage.measure

#The abstract class of a node in a network of agents

class Node:
    def __init__(self, index):
        self.index = index
        self.state = [] #The list of local state at different time steps
        self.action = [] #The list of local actions at different time steps
        self.reward = [] #The list of local actions at different time steps
        self.currentTimeStep = 0 #Record the current time step.
        self.paramsDict = {} #use a hash map to query the parameters given a state (or neighbors' states)
        self.QDict = {} #use a hash map to query to the Q value given a (state, action) pair
        self.kHop = [] #The list to record the (state, action) pairs of k-hop neighbors
    #get the local state at timeStep
    def getState(self, timeStep):
        if timeStep <= len(self.state) - 1:
            return self.state[timeStep]
        else:
            print("getState: In node ", self.index, ", timeStep overflows!")
            return -1
    #get the local action at timeStep
    def getAction(self, timeStep):
        if timeStep <= len(self.action) - 1:
            return self.action[timeStep]
        else:
            print("getAction: In node ", self.index, ", timeStep overflows!")
            return -1
    #get the local reward at timeStep
    def getReward(self, timeStep):
        if timeStep <= len(self.reward) - 1:
            return self.reward[timeStep]
        else:
            print("getReward: In node ", self.index, ", timeStep overflows!")
            return -1
    #get the kHopStateAction at timeStep
    def getKHopStateAction(self, timeStep):
        if timeStep <= len(self.kHop) - 1:
            return self.kHop[timeStep]
        else:
            print("getKHopStateAction: In node ", self.index, ", timeStep overflows!")
            return -1
    #get the local Q at timeStep
    def getQ(self, kHopStateAction):
        #if the Q value of kHopStateAction hasn't been queried before, return 0.0 (initial value)
        return self.QDict.get(kHopStateAction, 0.0)
    
    #initialize the local state
    def initializeState(self):
        pass
    #update the local state, it may depends on the states of other nodes at the last time step.
    #Remember to increase self.currentTimeStep by 1
    def updateState(self):
        pass
    #update the local action
    def updateAction(self):
        pass
    #update the local reward
    def updateReward(self):
        pass
    #update the local Q value
    def updateQ(self):
        pass
    #update the local parameter
    def updateParams(self):
        pass
    #clear the record. Called when a new inner loop starts. 
    def restart(self):
        self.state.clear()
        self.action.clear()
        self.reward.clear()
        self.kHop.clear()
        self.currentTimeStep = 0
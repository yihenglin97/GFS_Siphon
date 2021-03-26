import matplotlib.pyplot as plt
from tqdm import trange
import os
import numpy as np
import math
from scipy import special
import skimage.measure

# The following class defines a (grid) network of sensors.

class WeatherNetwork:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.nodeNum = height * width
        self.neighborDict = {} #use a hashmap to store the ((node, dist), neighbors) pairs which we have computed
    
    #convert the agent's index to coordinate
    def toCoordinate(self, i):
        if i >= 0 and i < self.nodeNum:
            h = i//self.width
            w = i%self.width
            return (h, w)
        else:
            return (-1, -1)
    
    #convert the agent's coordinate to index
    def toIndex(self, h, w):
        if h >= 0 and h < self.height and w >= 0 and w < self.width:
            return h * self.width + w
        else:
            return -1
    
    #query the d-hop neighborhood of node i, return a list of node indices.
    def findNeighbors(self, i, d):
        neighbors = []
        if i >= self.nodeNum or i < 0:
            print("Invalid node index!")
            return neighbors
        if (i, d) in self.neighborDict: #if we have computed the answer before, return it
            return self.neighborDict[(i, d)]
        h, w = self.toCoordinate(i)
        neighbor_candidates = []
        for j1 in range(-d, d+1):
            remain = d - abs(j1)
            for j2 in range(-remain, remain + 1):
                neighbor_candidates.append((h + j1, w + j2))
        for candidate in neighbor_candidates:
            candidate_index = self.toIndex(candidate[0], candidate[1])
            if candidate_index >= 0:
                neighbors.append(candidate_index)
        self.neighborDict[(i, d)] = neighbors #cache the answer so we can reuse later
        return neighbors

#This function describes how the energy level evolves:

def energy_dynamics(energy_level, disturbance):
    max_energy_level = 1
    min_energy_level = 0
    new_energy_level = int(energy_level + disturbance)
    if new_energy_level > max_energy_level:
        new_energy_level = max_energy_level
    elif new_energy_level < min_energy_level:
        new_energy_level = min_energy_level
    return new_energy_level

#The following class defines how the MDP environment evolves.

class WeatherGridEnv:
    def __init__(self, height, width, num_state, thres, e_thres, reward, dataset, e_dataset, T):
        self.height = height
        self.width = width
        self.nodeNum = self.height * self.width
        self.num_state = num_state
        self.dataset = dataset
        self.e_dataset = e_dataset
        self.T = T
        self.vmin = np.amin(dataset)
        self.vmax = np.amax(dataset)
        self.e_vmin = np.amin(e_dataset)
        self.e_vmax = np.amax(e_dataset)
        self.threshold = thres
        self.e_threshold = e_thres
        self.observationReward = reward #reward for observing a difference
        
        #some default parameters
        self.measureCost = 0.1 #cost for taking a measurement
        
        self.hiddenSeq = None #will be used to store the randomly sampled trajectory
        self.e_hiddenSeq = None 
        self.t = 0
        
        self.weatherNetwork = WeatherNetwork(self.height, self.width)
        self.globalState = np.zeros((self.nodeNum, 2), dtype=int)
        self.newGlobalState = self.globalState.copy()
        self.globalAction = np.zeros(self.nodeNum, dtype=int)
        self.globalReward = np.zeros(self.nodeNum, dtype = float)
    
    #convert measurement to state by equally divide the measurement range
    def toState(self, measurement, vmin, vmax, num_state):
        divideLine = np.linspace(start = vmin, stop = vmax, num = num_state + 1)
        state = np.zeros((self.height, self.width), dtype = int)
        for i in range(self.height):
            for j in range(self.width):
                for q in range(1, self.num_state + 1):
                    if measurement[i, j] >= divideLine[self.num_state - q]:
                        state[i, j] = self.num_state - q
                        break
        return state
    
    def toEState(self, measurement):
        e_state = np.zeros((self.height, self.width), dtype = int)
        for i in range(self.height):
            for j in range(self.width):
                if measurement[i, j] >= self.e_threshold:
                    e_state[i, j] = 1
        return e_state
                    
    #initialize the environment by randomly select the starting time
    def initialize(self):
        L, _, _ = self.dataset.shape
        t0 = int(np.random.uniform(low = 1e-4, high = L - self.T - 1e-4))
        self.hiddenSeq = self.dataset[t0: t0 + self.T, :, :]
        self.e_hiddenSeq = self.e_dataset[t0: t0 + self.T, :, :]
        self.t = 0
        
        #the 0 th column is the measurement state
        self.globalState[:, 0] = np.reshape(self.toState(self.hiddenSeq[0, :, :], self.vmin, self.vmax, self.num_state), self.nodeNum)
        #the 1 th column is the energy state       
        self.globalState[:, 1] = np.ones(self.nodeNum)
        #np.reshape(self.toState(self.e_hiddenSeq[0, :, :], self.e_vmin, self.e_vmax, 3), self.nodeNum)
        self.globalReward = np.zeros(self.nodeNum, dtype = float)
    
    #query the states of depth-hop neighbors
    def observeStateG(self, index, depth):
        result = []
        for j in self.weatherNetwork.findNeighbors(index, depth):
            result.append(tuple(self.globalState[j, :]))
        return tuple(result)
    
    def observeStateActionG(self, index, depth):
        result = []
        for j in self.weatherNetwork.findNeighbors(index, depth):
            result.append((tuple(self.globalState[j, :]), self.globalAction[j]))
        return tuple(result)
    
    def observeReward(self, index):
        return self.globalReward[index]
    
    #assign reward based on current actions and measurement
    def generateReward(self):
        measurement = self.hiddenSeq[self.t, :, :]
        
        #for each agent, if its measurement is far from one of its neighbors, it will receive a reward of 1
        for index in range(self.nodeNum):
            reward = 0.0
            #the agent needs at least one unit of energy to do the measurement
            if self.globalAction[index] == 1 and self.globalState[index, 1] >= 1:
                reward -= self.measureCost
                h, w = self.weatherNetwork.toCoordinate(index)
                outcome = measurement[h, w]
                neighbors = self.weatherNetwork.findNeighbors(index, 1)
                for neighbor in neighbors:
                    #check if the neighbor successfully did the measurement
                    if self.globalAction[neighbor] == 1 and self.globalState[neighbor, 1] >= 1:
                        h2, w2 = self.weatherNetwork.toCoordinate(neighbor)
                        neighbor_outcome = measurement[h2, w2]
                        if abs(neighbor_outcome - outcome) >= self.threshold:
                            reward += self.observationReward
                            break
            self.globalReward[index] = reward
        
        self.newGlobalState[:, 0] = np.reshape(self.toState(self.hiddenSeq[self.t + 1, :, :], self.vmin, self.vmax, self.num_state), self.nodeNum)
        disturbance = - self.globalAction + np.reshape(self.toEState(self.e_hiddenSeq[self.t + 1, :, :]), self.nodeNum)
        for index in range(self.nodeNum):
            #if np.random.uniform() >= 0.3:
            #    disturbance[index] += 1
            self.newGlobalState[index, 1] = energy_dynamics(self.globalState[index, 1], disturbance[index])
        
    def step(self):
        self.globalState = self.newGlobalState
    
    def updateAction(self, index, action):
        self.globalAction[index] = action
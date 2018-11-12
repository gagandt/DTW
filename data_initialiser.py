import numpy as np
import pandas as pd
import sys
import os

from dtw_super import DTWSuper

#Data initialiser class.
#Initialises the data given the value of M.
#stores data in the form of np-arrays.

class DATA:
    M = ""
    x_test = np.array([])
    y_test = np.array([])
    x1_train = np.array([])
    x2_train = np.array([])
    x3_train = np.array([])
    y1_train = np.array([])
    y2_train = np.array([])
    y3_train = np.array([])
    k = 4
    
    def __init__(self, m_value, k_value):
        self.M = str(m_value)
        self.training(self.M)
        self.testing(self.M)
        self.k = k_value
    
    def training(self, M):
        x1 = []
        x2 = []
        x3 = []
        
        with open('data/'+M+'/train/x1_train.txt') as f:
           for line in f:
               seq = [int(x) for x in line.split()]
               x1.append(seq)
        
        with open('data/'+M+'/train/x2_train.txt') as f:
           for line in f:
               seq = [int(x) for x in line.split()]
               x2.append(seq)
        
        with open('data/'+M+'/train/x3_train.txt') as f:
           for line in f:
               seq = [int(x) for x in line.split()]
               x3.append(seq)
        
        self.x1_train = np.array(x1)
        self.x2_train = np.array(x2)
        self.x3_train = np.array(x3)
        
        self.y1_train = np.full(len(self.x1_train), 1)
        self.y2_train = np.full(len(self.x2_train), 2)
        self.y3_train = np.full(len(self.x3_train), 3)
        
    def testing(self, M):
        x = []
        y = []
        
        with open('data/'+M+'/test/x1_test.txt') as f:
           for line in f:
               seq = [int(x) for x in line.split()]
               x.append(seq)
               y.append(1)
        
        
        with open('data/'+M+'/test/x2_test.txt') as f:
           for line in f:
               seq = [int(x) for x in line.split()]
               x.append(seq)
               y.append(2)
        
        
        with open('data/'+M+'/test/x3_test.txt') as f:
           for line in f:
               seq = [int(x) for x in line.split()]
               x.append(seq)
               y.append(3)
        
        self.x_test = np.array(x)
        self.y_test = np.array(y)
        
    def fit(self):
        model = DTWSuper(self.x_test, self.y_test, self.x1_train, self.x2_train, self.x3_train, self.y1_train, self.y2_train, self.y3_train, self.k)
        return model.knn_classifier()


        
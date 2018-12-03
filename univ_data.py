import numpy as np
import sys
import os
import glob
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import euclidean

class UNIVdata:
    x_test = np.array([])
    y_test = np.array([])
    x1_train = np.array([])
    x2_train = np.array([])
    x3_train = np.array([])
    y1_train = np.array([])
    y2_train = np.array([])
    y3_train = np.array([])
    sound = ""
    
    def __init__(self, s):
        self.training()
        
    def training(self):
        x1 = []
        x2 = []
        x3 = []
        
        for filename in os.listdir('./DATA/Train/re/'):
            with open('./DATA/Train/re/' + filename) as f:
                seq = [[float(x) for x in line.split()] for line in f]
                x1.append(seq)
        for filename in os.listdir('./DATA/Train/ri/'):
            with open('./DATA/Train/ri/' + filename) as f:
                seq = [[float(x) for x in line.split()] for line in f]
                x2.append(seq)
        
        for filename in os.listdir('./DATA/Train/rI/'):
            with open('./DATA/Train/rI/' + filename) as f:
                seq = [[float(x) for x in line.split()] for line in f]
                x3.append(seq)
        
        print(len(x1))
        print(len(x2))
        print(len(x3))
        
        
    
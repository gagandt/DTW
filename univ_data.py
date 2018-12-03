import numpy as np
import sys
import os
import glob
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import euclidean


from dtw_super import DTWSuper

class UNIVdata:
    x_test = np.array([])
    y_test = np.array([])
    x1_train = np.array([])
    x2_train = np.array([])
    x3_train = np.array([])
    y1_train = np.array([])
    y2_train = np.array([])
    y3_train = np.array([])
    k = 4
    
    def __init__(self, kval):
        self.training()
        self.testing()
        self.k = kval
        
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
        
        self.x1_train = np.array(x1)
        self.x2_train = np.array(x2)
        self.x3_train = np.array(x3)
        
        self.y1_train = np.full(len(self.x1_train), 1)
        self.y2_train = np.full(len(self.x2_train), 2)
        self.y3_train = np.full(len(self.x3_train), 3)
        
        
    def testing(self):
        x = []
        y = []
        
        for filename in os.listdir('./DATA/Test/re/'):
            with open('./DATA/Test/re/' + filename) as f:
                seq = [[float(x) for x in line.split()] for line in f]
                x.append(seq)
                y.append(1)
        
        for filename in os.listdir('./DATA/Test/ri/'):
            with open('./DATA/Test/ri/' + filename) as f:
                seq = [[float(x) for x in line.split()] for line in f]
                x.append(seq)
                y.append(2)
        
        for filename in os.listdir('./DATA/Test/rI/'):
            with open('./DATA/Test/rI/' + filename) as f:
                seq = [[float(x) for x in line.split()] for line in f]
                x.append(seq)
                y.append(3)
        
        self.x_test = np.array(x)
        self.y_test = np.array(y)
        
    def fit(self):
        model = DTWSuper(self.x_test, self.y_test, self.x1_train, self.x2_train, self.x3_train, self.y1_train, self.y2_train, self.y3_train, self.k)
        return model.knn_classifier()
        
        
    
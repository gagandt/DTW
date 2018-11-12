import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

from dtw_super import DTWSuper
from data_initialiser import DATA
#Number of Nearest Neighbours.
f = 4;

a1 = []
a2 = []
a3 = []

for i in range(1, 5):
    #Initialising data for different Observation Symbol Numbers. 
    d1 = DATA(8,f)
    d2 = DATA(16,f)
    d3 = DATA(32,f)

    print("for k = 8, KNN k  = " + str(f))
    a1.append(d1.fit())
    print("for k = 16, KNN k  = " + str(f))
    a2.append(d2.fit())
    print("for k = 32, KNN k  = " + str(f))
    a3.append(d3.fit())
    f *= 2

#Plotting the Accuracy Vs Nearest Neighbours Curve.
x = [4,8,16,32]
A = [a1, a2, a3]
kk = 4
for i in range(0, 3):
    kk *= 2
    plt.scatter(x, A[i])
    plt.xlabel("Nearest Neighbours")
    plt.ylabel("Accuracy")
    plt.title("Plot for Accuracy Vs 'K' in KNN for and no. of observation symbols = " + str(kk))
    
    xtic = [4, 8,16,32]
    xlab = ['4', '8', '16', '32']
    
    plt.xticks(xtic, xtic)
    #plt.yticks(a1,a1)
    
    plt.show()

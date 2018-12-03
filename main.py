import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

#from dtw_super import DTWSuper
from univ_data import UNIVdata
#from data_initialiser import DATA
#Number of Nearest Neighbours.



d1 = UNIVdata(4)
print(d1.fit())

d1 = UNIVdata(8)
print(d1.fit())

d1 = UNIVdata(16)
print(d1.fit())

d1 = UNIVdata(32)
print(d1.fit())



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import randint
import DiscreteCondEnt as DC

low, high, RVsize, numRV = 0, 1, 1000, 6
depend = np.array([0, 1, 3])
rv = DC.getRandomVar_select(np.random.uniform, low, high, RVsize, numRV, depend)

from sklearn.metrics import mean_squared_log_error
def logMSEscorer(clf, X, y):
    y_est = clf.predict(X)
    return mean_squared_log_error(y, y_est)

# from sklearn.tree import DecisionTreeRegressor
# regressor = DecisionTreeRegressor(random_state=0)

from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=numRV)

CVFold = 3
DC.computeEnt(rv, regressor,logMSEscorer, CVFold)
# histRV = np.array(np.unique(rv[0], axis=0, return_counts=True)[1])[None,:]
# histRV2 = np.array(np.unique(rv[(0,1),:], axis=1, return_counts=True)[1])[None,:]
# for i in range(1, numRV):
#     histRV = np.append(histRV, np.array(np.unique(rv[i], axis=0, return_counts=True)[1])[None,:], axis=0)
#     if i!=1:
#         for j in range(i):
#             newArr = np.unique(rv[(j,i),:], axis=1, return_counts=True)
#             histRV2 = np.append(histRV2, np.array(newArr[1])[None,:], axis=0)
# pmfRV = histRV/RVsize
# pmfRV2 = histRV2/RVsize
# H = -1*np.average(np.log(pmfRV), weights=pmfRV, axis=1)
# H2 = -1*np.average(np.log(pmfRV2), weights=pmfRV2, axis=1)
# MI = np.zeros(H2.shape)
# index = 0
# for i in range(1, numRV):
#     for j in range(i):
#         MI[index] = H2[index] - H[i] - H[j]
#         index = index + 1

#print(MI)


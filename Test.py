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

def varEntropy(y):
    return -1*np.log(np.var(y))

# from sklearn.tree import DecisionTreeRegressor
# regressor = DecisionTreeRegressor(random_state=0)

from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=numRV)

CVFold = 3
DC.computeEnt(rv, regressor,logMSEscorer, varEntropy, CVFold)

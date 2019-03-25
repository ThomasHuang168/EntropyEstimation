import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import randint

low, high, RVsize, numRV = 0, 2, 1000, 4
CVFold = 3


rv = np.split(randint.rvs(low, high, size=RVsize*(numRV - 1)), numRV - 1)
rv = np.append(rv, np.remainder(np.sum(rv, axis=0), high)[None,:], axis=0)
histRV = np.array(np.unique(rv[0], axis=0, return_counts=True)[1])[None,:]
histRV2 = np.array(np.unique(rv[(0,1),:], axis=1, return_counts=True)[1])[None,:]
for i in range(1, numRV):
    histRV = np.append(histRV, np.array(np.unique(rv[i], axis=0, return_counts=True)[1])[None,:], axis=0)
    if i!=1:
        for j in range(i):
            newArr = np.unique(rv[(j,i),:], axis=1, return_counts=True)
            histRV2 = np.append(histRV2, np.array(newArr[1])[None,:], axis=0)


pmfRV = histRV/RVsize
pmfRV2 = histRV2/RVsize
H = -1*np.average(np.log(pmfRV), weights=pmfRV, axis=1)
H2 = -1*np.average(np.log(pmfRV2), weights=pmfRV2, axis=1)
MI = np.zeros(H2.shape)
index = 0
for i in range(1, numRV):
    for j in range(i):
        MI[index] = H2[index] - H[i] - H[j]
        index = index + 1

from scipy.special import comb
def subset(size, index):
    subset = [-1]
    sum = 0
    for numOutput in range(size + 1):
        c = comb(size, numOutput)
        if index >= sum + c:
            sum += c
        else:
            break
    #print (numOutput)
    numLeft = numOutput
    for candidate in range(size-1, -1, -1):
        if index == sum:
            for remaining in range(numLeft-1, -1, -1):
                if subset[0] == -1:
                    subset[0] = remaining
                else:
                    subset = np.append(subset, remaining)
            break
        elif 0 == numLeft:
            break
        elif (index - sum) >= comb(candidate, numLeft):
            sum += comb(candidate, numLeft)
            if subset[0] == -1:
                subset[0] = candidate
            else:
                subset = np.append(subset, candidate)
            numLeft -= 1
    #print(output)
    if subset[0] != -1:
        return subset

def ConditionSet(size, Resp, index):
    set = subset(size - 1, index)
    cond = [-1]
    for element in set:
        if element >= Resp:
            element += 1
        if cond[0] == -1:
            cond[0] = element
        else:
            cond = np.append(cond, element)
    return cond

def DiscreteEntropy(y):
    #cols = y.shape[y.ndim-1]
    #rows = y.shape[0]
    pmf = np.unique(y, return_counts=True, axis=y.ndim-1)[1]/y.shape[y.ndim-1]
    return -1*np.average(np.log(pmf), weights=pmf)

def CondDEntropyScorer(estimator, X, y):
    y_est = estimator.predict(X)
    #print (np.unique(np.array([y,y_est]), return_counts=True, axis=1))
    return DiscreteEntropy(np.array([y,y_est])) - DiscreteEntropy(y_est)



'''
CART
'''
from sklearn import tree
clf = tree.DecisionTreeClassifier() #Good for high==2


'''
SVM
'''
#from sklearn import svm
#clf = svm.SVC(gamma='scale', decision_function_shape='ovo') #Not sure

'''
KNN
'''
# from sklearn import neighbors
# clf = neighbors.NearestCentroid() #Not sure
# numNeighbors = high
# clf = neighbors.KNeighborsClassifier(numNeighbors) #better than CART


from sklearn.model_selection import cross_val_score
#print (cross_val_score(clf,np.transpose(rv[ConditionSet(numRV, 0, 6)]), rv[0], cv=3, scoring=CondEntropyScorer))

numComb = np.power(2, numRV - 1)
DEntropy = np.zeros((numRV, numComb))
print ("Discrete RV with range [", low, ", ", high, ")")
for Resp in range(numRV):
    DEntropy[Resp,0] = DiscreteEntropy(rv[Resp])
    for sI in range(1, numComb):
        DEntropy[Resp,sI] = np.mean(cross_val_score(clf,np.transpose(rv[ConditionSet(numRV, Resp, sI)]), rv[Resp], cv=CVFold, scoring=CondDEntropyScorer))
        print ("Under cond=", ConditionSet(numRV, Resp, sI), "\t= ", DEntropy[Resp,sI])
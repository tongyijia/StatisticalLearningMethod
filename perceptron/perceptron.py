# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 17:56:17 2019

@author: 同乐乐
"""

import numpy as np
import time

def loadData(filename):
    
    print('start to load data')
    
    dataArr = []
    dataLable = []
    
    f = open(filename, 'r')
    
    for line in f.readlines():
        curLine = line.strip().split(',')
        
        if int(curLine[0]) >= 5:
            dataLable.append(1)
        else:
            dataLable.append(-1)
            
        dataArr.append([int(num)/255 for num in curLine[1:]])
    
    return dataArr, dataLable


def perceptron(dataArr, lableArr, iter):    
    
    print('start to train')
    
    dataMat = np.mat(dataArr)
    lableMat = np.mat(lableArr).T
    
    m, n = np.shape(dataMat)
    w = np.zeros((1, np.shape(dataMat)[1]))
    b = 0
    h = 0.0001
    
    for j in range(iter):
        for i in range(m):
            xi, yi = dataMat[i], lableMat[i]
            if yi * ( w * xi.T + b) <= 0:
                w = w + h * yi * xi
                b = b + h * yi
    
        print('Round %d : %d training' % (j, iter))
        
    return w, b
    
def test(dataArr, lableArr, w, b):
    
    dataMat = np.mat(dataArr)
    lableMat = np.mat(lableArr).T
    
    m, n = np.shape(dataMat)
    errsum = 0
    
    for i in range(m):
        
        xi, yi = dataMat[i], lableMat[i]
        
        if (w * xi.T + b) * yi <= 0:
            errsum += 1
            
    accruRate = 1 - (errsum / m)
            
    return accruRate


if __name__ == '__main__':
    
    start = time.time()
    
    trainData, trainLable = loadData('../Mnist/mnist_train.csv')
    testData, testLable = loadData('../Mnist/mnist_test.csv')
    
    w, b = perceptron(trainData, trainLable, iter = 30)
    accruRate = test(testData, testLable, w, b)
    
    end = time.time()    
    
    print('accruRate:', accruRate)
    
    print('time spent : ', end - start)
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:26:13 2019

@author: 同乐乐
"""

import numpy as np
import time


def loadData(filename):
    
    data = [];label = []
    
    fr = open(filename,'r')
    
    for line in fr.readlines():
        curLine = line.strip().split(',')
        
        if int(curLine[0]) == 0:
            label.append(1)
        else:
            label.append(0)
        
        data.append([int(num)/255 for num in curLine[1:]])
        
    return data, label

def logisticRegression(trainData, trainLabel, iter = 200):
    
    for i in range(len(trainLabel)):
        trainData[i].append(1)
    
    trainDataArr = np.array(trainData)
    
    w = np.zeros(trainDataArr.shape[1])
    
    h = 0.001
    
    for i in range(iter):
        #print('%d : %d'%(i, iter))
        for j in range(len(trainLabel)):
            
            wx = np.dot(w, trainDataArr[j])
            yi = trainLabel[j]
            xi = trainDataArr[j]
            
            w += h * (xi * yi -  (np.exp(wx) * xi) / (1 + np.exp(wx)))
    
    return w

def predict(testDataList, w):
    
    wx = np.dot(w, testDataList)
    
    P1 = np.exp(wx) / (1 + np.exp(wx))
    
    if P1 >= 0.5:
        return 1
    else:
        return 0
    
def test(testData, testLabel, w):
    
    errorCnt = 0
    
    for i in range(len(testLabel)):
        
        testData[i].append(1)
        
        if testLabel[i] != predict(testData[i], w):
            errorCnt += 1
    
    return 1 - (errorCnt / len(testLabel))
   

if __name__ == '__main__':
    
    start = time.time()
    
    print('start to load data')
    trainData, trainLabel = loadData('../Mnist/mnist_train.csv')
    testData, testLabel = loadData('../Mnist/mnist_test.csv')
    
    print('start to train')
    w = logisticRegression(trainData, trainLabel)
    
    print('start to test')
    accur = test(testData, testLabel, w)
    
    print('the accur is',accur)
    print('time spent',time.time() - start)
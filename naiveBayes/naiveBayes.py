# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 22:09:40 2019

@author: 同乐乐
"""

import numpy as  np
import time

def loadData(filename):
    
    print('start to load data...')
    
    DataArr = []; LableArr = []
    
    fr = open(filename, 'r')
    
    for line in fr.readlines():
        
        curLine = line.strip().split(',')
        
        LableArr.append(int(curLine[0]))
        DataArr.append([int(int(num) > 128) for num in curLine[1:]])

    return DataArr, LableArr

def getAllProbability(trainData, trainLable):
    
    featureNum = 784
    classNum = 10
    
    Py = np.zeros((classNum, 1))
    
    for i in range(classNum):
        
        Py[i] = np.sum(np.mat(trainLable) == i)
  
    Px_y = np.zeros((featureNum, 2, classNum))
    
    for i in range(len(trainLable)):
        x = trainData[i]
        y = trainLable[i]
        
        for j in range(featureNum):
            Px_y[j][x[j]][y] += 1
            
    
    for feature in range(featureNum):
        for y in range(classNum):
            Px_y[feature][0][y] = np.log((Px_y[feature][0][y] + 1) / (Py[y] + 1 * 2))            
            Px_y[feature][1][y] = np.log((Px_y[feature][0][y] + 1) / (Py[y] + 1 * 2))
        
            
    for i in range(classNum):
        Py[i] = (Py[i] + 1) / (len(trainLable) + classNum * 1)
        
    Py = np.log(Py)   
    
    return Py, Px_y
    

def naiveBayes(Py, Px_y, x):

    classNum = 10
    feature = 784
    
    predict = [0] * classNum
    
    for i in range(classNum):
        sum = 0
        for j in range(feature):
            sum += Px_y[j][x[j]][i]
        predict[i] = sum + Py[i]
    
    return predict.index(max(predict))
  

def test(Py, Px_y, testDataArr, testLableArr):
    
    errorCnt = 0
    
    for i in range(len(testDataArr)):
        
        presict = naiveBayes(Py, Px_y, testDataArr[i])
       
        if presict != testLableArr[i]:
           
            errorCnt += 1
    
    return 1 - (errorCnt / len(testDataArr))


if __name__ == '__main__':
    
    start = time.time()
    
    trainData, trainLable = loadData('../Mnist/mnist_train.csv')
    testData, testLable = loadData('../Mnist/mnist_test.csv')
    
    print('start to train...')
    Py, Px_y = getAllProbability(trainData, trainLable)
    
    print('start to test...')
    accuracy = test(Py, Px_y, testData, testLable)
    
    end = time.time()
    print('the accuracy is ', accuracy)
    print('time span', end - start)
    
    
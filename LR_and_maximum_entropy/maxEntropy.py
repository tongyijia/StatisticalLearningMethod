# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 23:35:30 2019

@author: 同乐乐
"""

import numpy as np
import time
from collections import defaultdict

def loadData(filename):
    
    dataList = []; labelList = []
    
    fr = open(filename, 'r')
    
    for line in fr.readlines():
        curLine = line.strip().split(',')
        if int(curLine[0]) == 0:
            labelList.append(0)
        else:
            labelList.append(1)
        
        dataList.append([int(int(num) > 128) for num in curLine[1:]])
    
    return dataList, labelList

class maxEnt:
    
    def __init__(self, trainDataList, trainLabelList, testDataList, testLabelList):
        
        self.trainDataList = trainDataList
        self.trainLabelList = trainLabelList
        self.testDataList = testDataList
        self.testLabelList = testLabelList
        self.featureNum = len(trainDataList[0])
        
        self.N = len(trainLabelList)
        self.n = 0                       #训练集中（xi，y）对数量
        self.M = 10000
        self.fixy = self.calc_fixy()    #所有（x，y）对出现的次数
        self.w = [0] * self.n
        self.xy2idDict, self.id2xyDict = self.createSearchDict()
        self.Ep_xy = self.calcEp_xy()
        
    def calcEpxy(self):
        
        Epxy = [0] * self.n 
        
        for i in range(self.N):
            
            Pwxy = [0] * 2
            
            Pwxy[0] = self.calcPwy_x(self.trainDataList[i], 0)
            Pwxy[1] = self.calcPwy_x(self.trainDataList[i], 1)
            
            for feature in range(self.featureNum):
                for y in range(2):
                    if (self.trainDataList[i][feature], y) in self.fixy[feature]:
                        id = self.xy2idDict[feature][(self.trainDataList[i][feature], y)]
                        Epxy[id] += (1 / self.N) * Pwxy[y]
            
        return Epxy
                        
                        
    def calcPwy_x(self, X, y):
        
        numerator = 0
        
        Z = 0
        
        for i in range(self.featureNum):
            if (X[i], y) in self.xy2idDict[i]:
                index = self.xy2idDict[i][(X[i], y)]
                numerator += self.w[index]
                
            if (X[i], 1-y) in self.xy2idDict[i]:
                index = self.xy2idDict[i][(X[i], 1-y)]
                Z += self.w[index]
        numerator = np.exp(numerator)
        Z = np.exp(Z) + numerator
        
        return numerator / Z
    
    def calcEp_xy(self):
        
        Ep_xy = [0] * self.n
        
        for feature in range(self.featureNum):
            
            for (x,y) in self.fixy[feature]:
                id = self.xy2idDict[feature][(x,y)]
                Ep_xy[id] = self.fixy[feature][(x,y)] / self.N
        
        return Ep_xy
                    
    def calc_fixy(self):
        
        fixyDict = [defaultdict(int) for i in range(self.featureNum)]
        
        for i in range(len(self.trainDataList)):
            for j in range(self.featureNum):
                fixyDict[j][(self.trainDataList[i][j], self.trainLabelList[i])] += 1
            
        for i in fixyDict:
            self.n += len(i)
            
        return fixyDict
    

    def createSearchDict(self):
        
        xy2idDict = [{} for i in range(self.featureNum)]
        
        id2xyDict = {}
        
        index = 0
        
        for feature in range(self.featureNum):
            for (x,y) in self.fixy[feature]:
                xy2idDict[feature][(x,y)] = index
                id2xyDict[index] = (x,y)
                index += 1
        return xy2idDict, id2xyDict
                        
    def maxEntropyTrain(self, iter = 500):
        
        for i in range(iter):
            print('%d : %d' % (i,iter))
            Epxy = self.calcEpxy()
            
            sigmaList = [0] * self.n
            
            for j in range(self.n):
                sigmaList[j] = (1 / self.M) * np.exp(self.Ep_xy[j] / Epxy[j])
            
            self.w = [self.w[i] + sigmaList[i] for i in range(self.n)]
            
    def predict(self, X):
        
        result = [0] * 2
        
        for i in range(2):
            result[i] = self.calcPwy_x(X, i)
            
        return result.index(max(result))

    def test(self):
        
        errCnt = 0
        
        for i in range(len(self.testLabelList)):
            result = self.predict(self.testDataList[i])
            
            if result != self.testLabelList[i]:
                errCnt += 1
        return 1 - (errCnt / len(self.testLabelList))


if __name__ == '__main__':
    
    start = time.time()
    
    print('start to load data...')
    trainData, trainLabel = loadData('../Mnist/mnist_train.csv')
    testData, testLabel = loadData('../Mnist/mnist_test.csv')
    
    maxEnt = maxEnt(trainData[:20000], trainLabel[:20000], testData, testLabel)
    
    print('start to train')
    maxEnt.maxEntropyTrain()
    
    print('start to test')
    accuracy = maxEnt.test()
    print('the accuracy is :', accuracy)
    
    print('time span:' , time.time() - start)
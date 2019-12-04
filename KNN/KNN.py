# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 09:20:34 2019

@author: 乐乐
"""

#%97

import numpy as np
import time

def loadData(filename):
    
    print('start to load data')
    
    dataArr = []
    dataLabel = []
    
    file = open(filename, 'r')
    
    for line in file.readlines():
         
        curLine = line.strip().split(',')
        
        dataLabel.append(int(curLine[0]))
        dataArr.append([int(num) for num in curLine[1:]])
        
    return dataArr,dataLabel

def getDist(x, y):
    
    dist = np.sqrt(np.sum(np.square(x - y)))
    
    return dist


    
def getCloest(trainDataMat, trainLabelMat, x, k):  
    
    distList = [0] * len(trainLabelMat)
    
    for i in range(len(trainLabelMat)):
        
        distList[i] = getDist(trainDataMat[i], x)
    
    #argsort：函数将数组的值从小到大排序后，并按照其相对应的索引值输出
    #例如：
    #   >>> x = np.array([3, 1, 2])
    #   >>> np.argsort(x)
    #   array([1, 2, 0])
    
    minKindex = np.argsort(np.array(distList))[:k]
    
    vote = [0] * 10
    
    for index in minKindex:
        
        vote[int(trainLabelMat[index])] += 1
        
    return vote.index(max(vote))

    

def test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, k):
    
    print('start to test...')
    
    trainDataMat = np.mat(trainDataArr); trainLabelMat = np.mat(trainLabelArr).T
    testDataMat = np.mat(testDataArr); testLabelMat = np.mat(testLabelArr).T
    
    errcount = 0
    
    for i in range(200):
        
        print('test %d:%d' % (i, 200))
        
        x = testDataMat[i]
        y = getCloest(trainDataMat, trainLabelMat, x, k)
        
        if y != testLabelMat[i]: errcount += 1
    
    return 1 - (errcount / 200)
        


if __name__ == '__main__':
    
    start = time.time()
    
    trainDataArr, trainLabelArr = loadData('../Mnist/mnist_train.csv')
    testDataArr, testLabelArr = loadData('../Minist/mnist_test.csv')
    
    accurate = test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, k=25)
    
    print('accurate is ',accurate)
    
    end = time.time()
    
    print('time:', end - start)
    
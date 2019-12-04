# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 22:34:51 2019

@author: 同乐乐
"""

'''
ID3未减枝
'''


import numpy as np
import time

def loadData(filename):
    
    dataArr = []
    labelArr = [] 
    
    fr = open(filename, 'r')
    
    for lines in fr.readlines():
        
        curLine = lines.strip().split(',')
        labelArr.append(int(curLine[0]))
        dataArr.append([int(int(num) > 128) for num in curLine[1:]])
    
    return dataArr, labelArr

def createTree(*dataset):
    
    Epsilon = 0.1
    
    trainDataList = dataset[0][0]
    trainLabelList = dataset[0][1]
    
    print('start a node ',len(trainDataList[0]), len(trainLabelList))
    
    classDict = {i for i in trainLabelList}
    
    if len(classDict) == 1:
        return trainLabelList[0]
    
    if len(trainDataList[0]) == 0:
        return majorClass(trainLabelList)
    
    Ag, EpsilonGet = calcBestFeature(trainDataList, trainLabelList)
    
    if EpsilonGet < Epsilon:
        return majorClass(trainLabelList)
    
    treeDict = {Ag:{}}
    
    treeDict[Ag][0] = createTree(getSubDataArr(trainDataList, trainLabelList, Ag, 0))
    treeDict[Ag][1] = createTree(getSubDataArr(trainDataList, trainLabelList, Ag, 1))
    
    return treeDict

def majorClass(labelList):
    
    classDict = {}
    
    for i in range(len(labelList)):
        if labelList[i] in classDict.keys():
            classDict[labelList[i]] += 1
        else:
            classDict[labelList[i]] = 1
    
    classSort = sorted(classDict.items(), key = lambda x : x[1], reverse=True)
    
    return classSort[0][0]

def calcBestFeature(dataList, labelList):
    
    dataArr = np.array(dataList)
    labelArr = np.array(labelList).T
    
    featureNum = dataArr.shape[1]
    
    maxG_D_A = -1
    maxFeature = -1
    
    for feature in range(featureNum):     
        H_D = calc_H_D(labelArr)
        
        trainDataArr_DevideByFeature = np.array(dataArr[:,feature].flat)
        
        G_D_A = H_D - calc_H_D_A(trainDataArr_DevideByFeature,labelArr)
        
        if G_D_A > maxG_D_A:
            maxG_D_A = G_D_A
            maxFeature = feature
    
    return maxFeature, maxG_D_A
    
def calc_H_D(labelArr):
    
    H_D = 0
    
    labelSet = set([i for i in labelArr])
    
    for i in labelSet:
        p = labelArr[labelArr == i].size / labelArr.size
        H_D += -1 * p * np.log2(p)
        
    return H_D
    
    
    
def calc_H_D_A(trainDataArr_DevideByFeature,labelArr):
    
    H_D_A = 0
    
    trainDataSet = set([label for label in trainDataArr_DevideByFeature])
    
    for i in trainDataSet:
        
        H_D_A += trainDataArr_DevideByFeature[trainDataArr_DevideByFeature == i].size / trainDataArr_DevideByFeature.size \
                 * calc_H_D(labelArr[trainDataArr_DevideByFeature == i])
    
    return H_D_A
        
    

def getSubDataArr(dataList, labelList, Ag, a):
    
    reDataList = []
    reLabelList = []
    
    for i in range(len(labelList)):
        if dataList[i][Ag] == a:
            reDataList.append(dataList[i][:Ag] + dataList[i][Ag+1:])
            reLabelList.append(labelList[i])
            
    return reDataList,reLabelList
    
    
def predict(dataList,tree):
    
    while True:
        
        (key, value), = tree.items()
        
        if type(tree[key]).__name__ == 'dict':
            
            dataVal = dataList[key]
            del dataList[key]
            tree = value[dataVal]
            
            if type(tree).__name__ == 'int':
                return tree
        else:
            return tree

def test(testDataArr, testLabelArr, tree):
    
    errcnt = 0
    
    for i in range(len(testLabelArr)):
        if testLabelArr[i] != predict(testDataArr[i],tree):
            errcnt += 1
    
    return 1 - (errcnt / len(testLabelArr))


if __name__ == '__main__':
    
    start = time.time()
    
    trainDataArr, trainLabelArr = loadData('../Mnist/mnist_train.csv')
    testDataArr, testLabelArr = loadData('../Mnist/mnist_test.csv')
    
    print('create tree ...')
    tree = createTree((trainDataArr, trainLabelArr))
    print('tree is ', tree)
    
    print('start to test ...')
    accur = test(testDataArr, testLabelArr, tree)
    print('the accuracy is ',accur)
    
    
    end = time.time()
    print('time spent ', end - start)
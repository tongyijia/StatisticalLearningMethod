# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 20:15:36 2019

@author: 同乐乐
"""

import time
import numpy as np

def loadData(filename):
    
    dataArr = []; labelArr = []

    fr = open(filename, 'r')
    
    for line in fr.readlines():
        
        curLine = line.strip().split(',')
        dataArr.append([int(int(num) > 128) for num in curLine[1:]])
        if int(curLine[0]) == 0:
            labelArr.append(1)
        else:
            labelArr.append(-1)
    
    return dataArr, labelArr

def calc_e_Gx(trainDataArr, trainLabelArr, n, div, rule, D):
    
    e = 0
    x = trainDataArr[:, n]
    y = trainLabelArr
    predict = []
    
    if rule == 'LisOne': L = 1; H = -1
    else:               L = -1; H = 1
    
    for i in range(trainDataArr.shape[0]):
        if x[i] < div:
            predict.append(L)
            if y[i] != L: e += D[i]
        elif x[i] >= div:
            predict.append(H)
            if y[i] != H: e += D[i]
            
    return np.array(predict), e
        

def createSigleBoostingTree(trainDataArr, trainLabelArr, D):
    
    m,n = np.shape(trainDataArr)
    sigleBoostTree = {}
    sigleBoostTree['e'] = 1
    
    for i in range(n):
        
        for div in [-0.5, 0.5, 1.5]:
            
            for rule in ['LisOne', 'HisOne']:
                Gx, e = calc_e_Gx(trainDataArr, trainLabelArr, i, div, rule, D)
                if e < sigleBoostTree['e']:
                    sigleBoostTree['e'] = e
                    sigleBoostTree['div'] = div
                    sigleBoostTree['rule'] = rule
                    sigleBoostTree['Gx'] = Gx
                    sigleBoostTree['feature'] = i
    
    return sigleBoostTree

def createBosstingTree(trainDataList, trainLabelList, treeNum = 50):
    
    trainDataArr = np.array(trainDataList)
    trainLabelArr = np.array(trainLabelList)
    
    #finalPredict = [0] * len(trainLabelList)
    m, n = np.shape(trainDataArr)
    
    D = [1 / m] * m
    
    tree = []
    
    for i in range(treeNum):
        curTree = createSigleBoostingTree(trainDataArr, trainLabelArr, D)
        alpha = 1 / 2 * np.log((1 - curTree['e']) / curTree['e'])
        Gx = curTree['Gx']
        D = np.multiply(D, np.exp(-1 *  alpha * np.multiply(trainLabelArr, Gx))) / sum(D)
        curTree['alpha'] = alpha
        tree.append(curTree)
        print('iter',i)
    return tree

def predict(x, div, rule, feature):
    if rule == 'LisOne':
        L = 1; H = -1
    else:
        L = -1; H = 1
        
    if x[feature] < div:
        return L
    else:
        return H
    

def test(testDataArr, testLabelArr, tree):
    errCnt = 0
    
    for i in range(len(testLabelArr)):
        result = 0
        for curTree in tree:
            div = curTree['div']
            rule = curTree['rule']
            feature = curTree['feature']
            alpha = curTree['alpha']
            
            result += alpha * predict(testDataArr[i], div, rule, feature)
        
        if np.sign(result) != testLabelArr[i]: errCnt += 1
    
    return 1 - (errCnt / len(testLabelArr))



if __name__ == '__main__':
    
    start = time.time()
    
    print('start to load data...')
    trainData, trainLabel = loadData('../Mnist/mnist_train.csv')
    testData, testLabel = loadData('../Mnist/mnist_train.csv')
    
    print('start to train...')
    tree = createBosstingTree(trainData[:10000], trainLabel[:10000], 40)
    
    print('start to test...')
    accuracy = test(testData[:1000], testLabel[:10000], tree)
    print('the accuracy is', accuracy)
    
    print('time span ',time.time() - start)
    
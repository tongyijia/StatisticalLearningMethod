# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:16:39 2019

@author: 同乐乐
"""

import time
import numpy as np
import math
import random

def loadData(filename):
    
    dataList = []; labelList = []
    
    fr = open(filename, 'r')
    
    for line in fr.readlines():
        curLine = line.strip().split(',')
        
        dataList.append([int(num) / 255 for num in curLine[1:]])
        
        if int(curLine[0]) == 0:
            labelList.append(1)
        else:
            labelList.append(-1)
    
    return dataList, labelList


class SVM:
    
    def __init__(self, trainDataList, trainLabelList, sigma = 10, C = 200, toler = 0.001):
        
        self.trainDataMat = np.mat(trainDataList)
        self.trainLabelMat = np.mat(trainLabelList).T
        
        self.sigma = sigma
        self.C = C
        self.toler = toler
        self.m, self.n = np.shape(self.trainDataMat)
        
        self.k = self.calcKernel()
        self.b = 0
        self.alpha = [0] * self.trainDataMat.shape[0]
        self.E = [0 * self.trainDataMat[i, 0] for i in range(self.trainLabelMat.shape[0])]   #
        self.supportVecIndex = []
        
        
    def calcKernel(self):
        
        k = [[ 0 for i in range(self.m)] for j in range(self.m)]
        
        for i in range(self.m):
            
            if i % 100 == 0:
                print('construct the kernel:', i, self.m)
            
            X = self.trainDataMat[i, :]
            
            
            for j in range(i, self.m):
                Z = self.trainDataMat[j, :]
                result = (X - Z) * (X - Z).T
                result = np.exp( -1 * result / (2 * self.sigma ** 2))
                k[i][j] = result
                k[j][i] = result
                
        
        return k
    
    def isSatisfyKKT(self, i):
        
        gxi = self.calc_gxi(i)
        yi = self.trainLabelMat[i]
        
        if (math.fabs(self.alpha[i]) < self.toler ) and (yi * gxi >= 1):
            return True
        elif (math.fabs(self.alpha[i] - self.C) < self.toler) and (yi * gxi <= 1):
            return True
        elif (self.alpha[i] > -self.toler) and (self.alpha[i] < (self.C + self.toler))\
                and (math.fabs(yi * gxi - 1) < self.toler):
            return True
        
        return False
    
    def calc_gxi(self, i):
        
        gxi = 0
        
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        
        for j in index:
            gxi += self.alpha[j] * self.trainLabelMat[j] * self.k[j][i]
            
        gxi += self.b
        
        return gxi
        
        
    def calcEi(self, i):
        
        gxi = self.calc_gxi(i)
        return gxi - self.trainLabelMat[i]
    
    
    def getAlphaJ(self, E1, i):
        
        E2 = 0
        maxE1_E2 = -1
        maxIndex = -1
        
        nozeroE = [i for i,Ei in enumerate(self.E) if Ei != 0]
        
        for j in nozeroE:
            E2_tmp = self.calcEi(j)
            if math.fabs(E1 - E2_tmp) > maxE1_E2:
                maxE1_E2 = math.fabs(E1 - E2_tmp)
                E2 = E2_tmp
                maxIndex = j
        
        if maxIndex == -1:
            maxIndex = i
            while maxIndex == i:
                maxIndex = int(random.uniform(0, self.m))
            E2 = self.calcEi(maxIndex)
            
        return E2, maxIndex
    
    def train(self, iter = 100):
        
        iterStep = 0; parameterChanged = 1
        
        while (iterStep < iter) and (parameterChanged > 0):
            
            print('iter:%d:%d'%( iterStep, iter))
            
            iterStep += 1
            parameterChanged = 0
            
            for i in range(self.m):
                
                if self.isSatisfyKKT(i) == False:
                    
                    E1 = self.calcEi(i)
                    E2, j = self.getAlphaJ(E1, i)
                    
                    y1 = self.trainLabelMat[i]
                    y2 = self.trainLabelMat[j]
                    
                   # print(" %d: %d - j: %d" %(y1, y2,j))
                    
                    alphaOld_1 = self.alpha[i]
                    alphaOld_2 = self.alpha[j]
                    
                    if y1 != y2:
                        L = max(0, alphaOld_2 - alphaOld_1)
                        H = min(self.C, self.C + alphaOld_2 - alphaOld_1)
                    else:
                        L = max(0, alphaOld_2 + alphaOld_1 - self.C)
                        H = min(self.C, alphaOld_2 + alphaOld_1)
                        
                    if L == H: 
                        continue
                
                    k11 = self.k[i][i]
                    k22 = self.k[j][j]
                    k21 = self.k[j][i]
                    k12 = self.k[i][j]
                    #print("k11",k11)
                    
                    #print("%d : %d : %d" %(k11,k22,k12))
                    alphaNew_2 = alphaOld_2 + y2 * (E1 - E2) / (k11 + k22 - 2 * k12)
                    
                    if alphaNew_2 < L: alphaNew_2 = L
                    elif alphaNew_2 > H: alphaNew_2 = H
                    
                    alphaNew_1 = alphaOld_1 + y1 * y2 * (alphaOld_2 - alphaNew_2)
                    
                    b1New = -1 * E1 - y1 * k11 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k21 * (alphaNew_2 - alphaOld_2) + self.b
                    b2New = -1 * E2 - y1 * k12 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k22 * (alphaNew_2 - alphaOld_2) + self.b
                            
                    if (alphaNew_1 > 0) and (alphaNew_1 < self.C):
                        bNew = b1New
                    elif (alphaNew_2 > 0) and (alphaNew_2 < self.C):
                        bNew = b2New
                    else:
                        bNew = (b1New + b2New) / 2
                        
                    self.alpha[i] = alphaNew_1
                    self.alpha[j] = alphaNew_2
                    self.b = bNew

                    self.E[i] = self.calcEi(i)
                    self.E[j] = self.calcEi(j)
                    
                    if math.fabs(alphaNew_2 - alphaOld_2) >= 0.00001:
                        parameterChanged += 1
                
               # print("iter: %d i:%d, pairs changed %d" % (iterStep, i, parameterChanged))
                        

        for i in range(self.m):
            #如果α>0，说明是支持向量
            if self.alpha[i] > 0:
                #将支持向量的索引保存起来
                self.supportVecIndex.append(i)
   
    def calcSingleKernel(self, X, Y):
        result = (X - Y) * (X - Y).T
        result = np.exp(-1 * result / (2 * self.sigma ** 2))
        
        return result
    
    def predict(self, X):
        
        result = 0
        
        for i in self.supportVecIndex:
            tmp = self.calcSingleKernel(self.trainDataMat[i, :], np.mat(X))
            result += self.alpha[i] * self.trainLabelMat[i] * tmp
        
        result += self.b
        
        return np.sign(result)
    
    def test(self, testDataList, testLabelList):
        
        errCnt = 0
        
        for i in range(len(testLabelList)):
            result = self.predict(testDataList[i])
            if result != testLabelList[i]:
                errCnt += 1
                
        return 1 - errCnt / len(testLabelList)

if __name__ == '__main__':
    
    start = time.time()
    
    print('start to load data...')
    trainData, trainLabel = loadData('../Mnist/mnist_train.csv')
    testData, testLabel = loadData('../Mnist/mnist_train.csv')
    
    svm = SVM(trainData[:1000], trainLabel[:1000], 10, 200, 0.001)
    
    print('start to train')
    svm.train()
    
    print('start to test')
    accuracy = svm.test(testData[:100], testLabel[:100])
    print('the accuracy is', accuracy)
    
    print('time span:', time.time() - start)
    
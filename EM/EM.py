# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 20:03:33 2019

@author: 同乐乐
"""

import time
import random
import numpy as np
import math

def loadData(mu0, sigma0, mu1, sigma1, alpha0, alpha1):
    
    length = 1000
    data0 = np.random.normal(mu0, sigma0, int(length * alpha0))
    data1 = np.random.normal(mu1, sigma1, int(length * alpha1))
    dataSet = []
    dataSet.extend(data0)
    dataSet.extend(data1)
    random.shuffle(dataSet)
    
    return dataSet

def calcGauss(dataSetArr, mu, sigma):
    
    result = 1 / (math.sqrt(2 * math.pi) * sigma) * \
            np.exp(-1 * (dataSetArr - mu) * (dataSetArr - mu) / (2 * sigma ** 2) )
    return result

def E_Step(dataSetArr, alpha0, mu0, sigma0, alpha1, mu1, sigma1):
    
    gamma0 = alpha0 * calcGauss(dataSetArr, mu0, sigma0)
    gamma1 = alpha1 * calcGauss(dataSetArr, mu1, sigma1)
    
    sum = gamma0 + gamma1
    
    gamma0 = gamma0 / sum
    gamma1 = gamma1 / sum
    
    return gamma0, gamma1
    
def M_Step(mu0, mu1, gamma0, gamma1, dataSetArr):
    
    mu0_new = np.dot(gamma0, dataSetArr) / np.sum(gamma0)
    mu1_new = np.dot(gamma1, dataSetArr) / np.sum(gamma1)

    sigma0_new = np.sum(np.dot(gamma0, (dataSetArr - mu0) ** 2)) / np.sum(gamma0)
    sigma1_new = np.sum(np.dot(gamma1, (dataSetArr - mu1) ** 2)) / np.sum(gamma1)
    
    alpha0 = np.sum(gamma0) / len(dataSetArr)
    alpha1 = np.sum(gamma1) / len(dataSetArr)
    
    return mu0_new, mu1_new, sigma0_new, sigma1_new, alpha0, alpha1
    
def EM_train(dataSetList, iter = 500):
    
    dataSetArr = np.array(dataSetList)
    
    alpha0 = 0.5; mu0 = 0; sigma0 = 1
    alpha1 = 0.5; mu1 = 1; sigma1 = 1
    
    step = 0
    
    while (step < iter):
        step += 1
        gamma0, gamma1 = E_Step(dataSetArr, alpha0, mu0, sigma0, alpha1, mu1, sigma1)
        mu0, mu1, sigma0, sigma1, alpha0, alpha1 = M_Step(mu0, mu1, gamma0, gamma1, dataSetArr)
        
    return alpha0, mu0, sigma0, alpha1, mu1, sigma1


if __name__ == '__main__':
    
    start = time.time()
    
    alpha0 = 0.3; mu0 = -2; sigmod0 = 0.5
    alpha1 = 0.7; mu1 = 0.5; sigmod1 = 1
    
    dataSetList = loadData(mu0, sigmod0, mu1, sigmod1, alpha0, alpha1)
    
    print('---------------------------')
    print('the Parameters set is:')
    print('alpha0:%.1f, mu0:%.1f, sigmod0:%.1f, alpha1:%.1f, mu1:%.1f, sigmod1:%.1f'%(
        alpha0, alpha1, mu0, mu1, sigmod0, sigmod1
    ))
    
    alpha0, mu0, sigmod0, alpha1, mu1, sigmod1 = EM_train(dataSetList)
    
    print('---------------------------')
    print('the Parameters predict is:')
    print('alpha0:%.1f, mu0:%.1f, sigmod0:%.1f, alpha1:%.1f, mu1:%.1f, sigmod1:%.1f'%(
        alpha0, alpha1, mu0, mu1, sigmod0, sigmod1
    ))
    
    print('time span:', time.time() - start)
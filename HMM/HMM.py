# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:24:00 2019

@author: 同乐乐
"""

import numpy as np
import time

def loadArticle(filename):
    
    artical = []
    fr = open(filename, encoding='utf-8')
    
    for line in fr.readlines():
        line = line.strip()
        artical.append(line)
        
    return artical

def trainParameter(filename):
    
    #定义一个查询字典，用于映射四种标记在数组中对应的位置，方便查询
    # B：词语的开头
    # M：一个词语的中间词
    # E：一个词语的结果
    # S：非词语，单个词
    statuDict = {'B':0, 'M':1, 'E':2, 'S':3}
    
    PI = np.zeros(4)
    A = np.zeros((4,4))
    B = np.zeros((4, 65536))
    
    fr = open(filename, encoding='utf-8')
    
    for line in fr.readlines():
        #---------------------训练集单行样例--------------------
        #深圳  有  个  打工者  阅览室
        #------------------------------------------------------
        #可以看到训练样本已经分词完毕，词语之间空格隔开，因此我们在生成统计时主要借助以下思路：
        # 1.先将句子按照空格隔开，例如例句中5个词语，隔开后变成一个长度为5的列表，每个元素为一个词语
        # 2.对每个词语长度进行判断：
        #       如果为1认为该词语是S，即单个字
        #       如果为2则第一个是B，表开头，第二个为E，表结束
        #       如果大于2，则第一个为B，最后一个为E，中间全部标为M，表中间词
        # 3.统计PI：该句第一个字的词性对应的PI中位置加1
        #           例如：PI = [0， 0， 0， 0]，当本行第一个字是B，即表示开头时，PI中B对应位置为0，
        #               则PI = [1， 0， 0， 0]，全部统计结束后，按照计数值再除以总数得到概率
        #   统计A：对状态链中位置t和t-1的状态进行统计，在矩阵中相应位置加1，全部结束后生成概率
        #   统计B：对于每个字的状态以及字内容，生成状态到字的发射计数，全部结束后生成概率
        #   注：可以看一下“10.1.1 隐马尔可夫模型的定义”一节中三个参数的定义，会有更清晰一点的认识
        #-------------------------------------------------------
        #对单行句子按空格进行切割
        curLine = line.strip().split()
        
        wordLabel = []
        
        for i in range(len(curLine)):
            if len(curLine[i]) == 1:
                label = 'S'
            else:
                label = 'B' + 'M' * (len(curLine[i]) - 2) + 'E'
        
            if i == 0:
                #如果是单行开头第一个字，PI中对应位置加1,
                PI[statuDict[label[0]]] += 1
            
            for j in range(len(label)):
                B[statuDict[label[j]]][ord(curLine[i][j])] += 1
                
            wordLabel.extend(label)
        
        for i in range(1, len(wordLabel)):
            A[statuDict[wordLabel[i - 1]]][statuDict[wordLabel[i]]] += 1
        
        
    sum = np.sum(PI)
    for i in range(len(PI)):
        if PI[i] == 0: PI[i] = -3.14e+100
        else: PI[i] = np.log(PI[i] / sum)
        
    for i in range(len(A)):
        sum = np.sum(A[i])
        for j in range(len(A[j])):
            if A[i][j] == 0: A[i][j] = -3.14e+100
            else: A[i][j] = np.log(A[i][j] / sum)
    
    for i in range(len(B)):
        sum = np.sum(B[i])
        for j in range(len(B[i])):
            if B[i][j] == 0: B[i][j] = -3.14e+100
            else:B[i][j] = np.log(B[i][j] / sum)
            
    return PI, A, B
    

def participle(artical, PI, A, B):
    
    retArtical =[]
    
    for line in artical:
        delta = [[0 for i in range(4)] for i in range(len(line))]
        
        for i in range(4):
            delta[0][i] = PI[i] + B[i][ord(line[0])]   ##??
        
        psi = [[0 for i in range(4)] for i in range(len(line))]
        
        for t in range(1, len(line)):
            for i in range(4):
                tmpDelta = [0] * 4
                for j in range(4):
                    tmpDelta[j] = delta[t - 1][j] + A[j][i]
                
                maxDelta = max(tmpDelta)
                maxDeltaIndex = tmpDelta.index(maxDelta)
                
                delta[t][i] = maxDelta + B[i][ord(line[t])]
                psi[t][i] = maxDeltaIndex
        
        sequence = []
        
        i_opt = delta[len(line) - 1].index(max(delta[len(line) - 1]))
        
        sequence.append(i_opt)
        
        for t in range(len(line) - 1, 0, -1):
            i_opt = psi[t][i_opt]
            sequence.append(i_opt)
        
        sequence.reverse()
            
        
        curLine = ''
        for i in range(len(line)):
            curLine += line[i]
            if (sequence[i] == 3 or sequence[i] == 2) and i != (len(line) - 1):
                curLine += '|'
                
        retArtical.append(curLine)
    return retArtical
            
        

if __name__ == '__main__':
    
    start = time.time()
    
    PI, A, B = trainParameter('HMMTrainSet.txt')
    
    artical = loadArticle('testArtical.txt')
    
    print('-------原文------')
    for line in artical:
        print(line)
        
    partiArtical = participle(artical, PI, A, B)
    
    print('-----分词后-----')
    for line in partiArtical:
        print(line)
        
    print('time span: ', time.time() - start)
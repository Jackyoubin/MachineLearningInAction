# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 13:30:54 2020

@author: lenovo
"""

import numpy as np
import operator

def creatDataset():
    group = np.array([[1,1.1],[1,1],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    '''
    k-近领算法
    
    ------------
    参数：
    inX：输入向量
    dataSet：输入样本集
    labels：标签向量
    k：选择最近邻的数目
    
    ------------
    返回：
    标签向量
    
    '''
    dataSetSize = dataSet.shape[0]
    difMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = difMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndecies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndecies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),\
                                 reverse=True)
    return sortedClassCount[0][0]

dataSet,labels = creatDataset()

l = classify0(np.array([[0,0]]),dataSet,labels,3)




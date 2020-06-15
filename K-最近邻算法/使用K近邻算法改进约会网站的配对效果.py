# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 14:59:53 2020

@author: lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import operator

#%%
'''
准备数据
'''
def file2matrix(filename):
    fr = open(filename)
    #readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表
    numberOfLines = len(fr.readlines())         
    returnMat = np.zeros((numberOfLines,3))        
    classLabelVector = []                         
    index = 0
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip()                             #去掉每行头尾空白
        listFromLine = line.split('\t')              #split() 通过指定分隔符对字符串进行切片
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    fr.close()
    return returnMat,classLabelVector

datingDataMat,datingLabels = file2matrix('D:\spyder_code\机器学习实战\K-最近邻算法\datingTestSet2.txt')







#%%
'''
不同列的差异过大，归一化处理
'''
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

normDataSet, ranges, minVals = autoNorm(datingDataMat)







#%%
'''
绘图
'''
def plotfig(normDataSet,datingLabels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    data = np.insert(normDataSet,3,np.array(datingLabels),axis=1)
    for m,i in [('o',1),('^',2),('*',3)]:
        dt = data[data[:,3]==i]
        xs = dt[:,0]
        ys = dt[:,1]
        zs = dt[:,2]
        ax.scatter(xs, ys, zs, marker=m)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()

plotfig(normDataSet,datingLabels)







#%%

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


'''
验证分类器
'''
def datingClassTest():
    hoRatio = 0.10      #hold out 10%
    datingDataMat,datingLabels = file2matrix('D:\spyder_code\机器学习实战\K-最近邻算法\datingTestSet2.txt')      
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): 
            errorCount += 1.0
    print ("the total error rate is: %f" % (errorCount/float(numTestVecs)))


datingClassTest()




















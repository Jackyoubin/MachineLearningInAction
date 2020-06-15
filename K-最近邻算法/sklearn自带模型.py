# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:22:40 2020

@author: lenovo
"""

from sklearn import neighbors
from sklearn import datasets


#%%

'''
虹膜数据集是经典且非常容易的多类别分类数据集
'''
iris = datasets.load_iris(return_X_y = False)

'''
K-n邻居分类KNeighborsClassifier 是最常用的技术。价值的最佳选择K高度依赖数据：通常情况下，  
 K抑制了噪声的影响，但使分类边界不那么明显
'''
knn=neighbors.KNeighborsClassifier()

knn.fit(iris.data,iris.target)

predictedLabel=knn.predict([[0.8, 1, 0.3, 0.5]])

print('predictedLabel is:%s' %predictedLabel )


        








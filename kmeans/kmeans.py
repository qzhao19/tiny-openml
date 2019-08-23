# -*- coding: utf-8 -*-

from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def loadData():
    file=open('dataSet2.txt')
    arrayOfLines=file.readlines()
    numLines=len(arrayOfLines)
    dataSet=zeros((numLines, 2), dtype=float)
    index=0
    for line in arrayOfLines:
        line=line.strip()
        listFromLine=line.split('\t')
        dataSet[index,:]=listFromLine[0:2]
        index+=1
    return dataSet

def distance(vectA, vectB):
    return sqrt(sum(power(vectA-vectB, 2)))

def randomCentroid(dataSet, k):
    dataSize=dataSet.shape[0]
    centroid=zeros((k,dataSet.shape[1]),dtype=float)
    index=random.randint(0,dataSize-1,k)
    centroid=dataSet[index,:]
    return centroid

def kmeans(dataSet,k):
    numLines=dataSet.shape[0]
    clusterAssment=zeros((numLines,2), dtype=float)     ###creer une matrice pour assigner les points da dataset
    centroid=randomCentroid(dataSet, k)
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        for i in range(numLines):       ###assigner chaque point pour le plus proche centroid
            minDist=inf; minIndex=-1
            for j in range(k):
                distEclud=distance(centroid[j,:], dataSet[i,:])     ###calculer la distance entre chaque point de centroids et celui de dataset
                if distEclud<minDist:                               ###mise a jour centroids
                    minDist=distEclud; minIndex=j
            if clusterAssment[i,0]!=minIndex:
                clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist**2
        print(centroid)
        for cent in range(k):           ###recalculer les centroids
            ptsInCluster=dataSet[nonzero(clusterAssment[:,0]==cent)[0]]
            ptsInCluster[cent,:]=mean(ptsInCluster,axis=0)
    return centroid, clusterAssment            
            

def showCluster(dataSet, k, centroids, clusterAssment):  
    numSamples, dim = dataSet.shape  
    if dim != 2:  
        print("Sorry! I can not draw because the dimension of your data is not 2!")
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
    if k > len(mark):  
        print("Sorry! Your k is too large! please contact Zouxy")
    # draw all samples  
    for i in range(numSamples):  
        markIndex = int(clusterAssment[i, 0])  
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])  
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  
    # draw the centroids  
    for i in range(k):  
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)  
    plt.show()  

k=3
dataSet=loadData()
centroid, clusterAssment=kmeans(dataSet,k)
showCluster(dataSet, k, centroid, clusterAssment)






































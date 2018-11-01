from math import log
import operator

def calcShannonEnt (dataSet):
    numEntries = len (dataSet)#计算数据集中实例总数
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] +=1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet ():
    dataSet = [[1,1, 'yes'],
               [1,1, 'yes'],
               [1,0, 'no'],
               [0,1, 'no'],
               [0,1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet,labels


#按照给定特征划分数据集
#三个输入参数分别为：待划分的数据集、划分数据集的特征、需要返回的特征的值
def splitDataSet (dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[: axis]
            reducedFeatVec.extend(featVec[axis+1 :])
            #print(reducedFeatVec)
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit (dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt (classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys() :
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),
      key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def createTree (dataSet, labels):
    classList = [example[-1] for example in dataSet]
    #类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #遍历完所有特征时返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeatLabel])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,
        bestFeat, value), subLabels)
    return myTree

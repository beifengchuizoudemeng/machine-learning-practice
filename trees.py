from math import log
import operator

#计算给定数据集的香农熵
def calcShannonEnt (dataSet):
    numEntries = len (dataSet)#计算数据集中实例总数
    #为所有可能分类创建字典
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

#创建测试用例
def createDataSet ():
    dataSet = [[1,1, 'yes'],
               [1,1, 'yes'],
               [1,0, 'no'],
               [0,1, 'no'],
               [0,1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet,labels


#按照给定特征划分数据集
#三个输入参数分别为：待划分的数据集dataSet、划分数据集的特征axis(第几个特征)、
#需要返回的特征的值value
def splitDataSet (dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[: axis]
            reducedFeatVec.extend(featVec[axis+1 :])
            #print(reducedFeatVec)
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的数据集划分方式
#假定数据是一种由列表元素组成的列表，且所有的列表元素具有相同的数据长度
#假定数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签
def chooseBestFeatureToSplit (dataSet):
    numFeatures = len(dataSet[0]) - 1#判定当前数据集包含多少特征属性
    baseEntropy = calcShannonEnt(dataSet)#计算划分前的信息熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        #创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)#set()函数创建一个无序不重复元素集。
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy#计算每个划分的信息增益
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#得到classList里面出现频率最高的类并返回分类名称
def majorityCnt (classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys() :
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),
      key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

#创建树的函数代码
#代码使用了两个输入参数：数据集dataSet和标签列表labels
def createTree (dataSet, labels):
    classList = [example[-1] for example in dataSet]#得到数据集所有类标签
    #递归停止条件一类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    #递归停止条件二遍历完所有特征时返回出现次数最多的类别
    if len(dataSet[0]) == 1:#长度为1，说明划分到了最后一个特征属性
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])#删除当前最好划分的特征
    #得到列表包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,
        bestFeat, value), subLabels)
    return myTree

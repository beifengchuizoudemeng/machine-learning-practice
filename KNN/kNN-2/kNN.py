from numpy import*
import os
import operator

def classify0 (inX, dataSet, labels, k):
    #计算距离并排序
    dataSetSize = dataSet.shape[0]#shape()返回的是维数的长度
    diffMat = tile(inX, (dataSetSize, 1)) -  dataSet#tile把inX在行上重复dataSetSize次，列上一次
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()#argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    #选择距离最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #计算频率并排序，返回频率最高的类别
    sortedClassCount = sorted(classCount.items(),#Python3.5中：iteritems变为items
     key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def img2vector (filename):
    returnVect = zeros((1,1024))
    fr = open (filename,'r')
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)

    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('trainingDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat,
        hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" %
        (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is: %d " % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/(mTest)))

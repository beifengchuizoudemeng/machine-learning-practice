from numpy import*
import operator

def createDataSet ():
    group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

#classify0()函数有4个参数：用于分类的输入向量in,输入的训练样本集dataSet,
#标签向量labels,最后k表示用于选择最近邻居的数目
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

#将文本记录转换为NumPy的解析函数
def file2matrix(filename):
    fr = open (filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

#归一化特征值,目的是计算距离时，每个特征值影响相同，不同时可以相应的加权
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet / tile(ranges, (m,1))
    return normDataSet, ranges, minVals

#分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int (m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult =  classify0(normMat[i,:],normMat[numTestVecs:m,:],
        datingLabels[numTestVecs:m],11)
        print("the classifier came back with: %d, the real answer is: %d"%
        (classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input
    ("percentage of time spent playing video games?"))
    ffMiles =  float(input
    ("frequent flier miles earned per year?"))
    iceCream = float(input
    ("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,
    normMat,datingLabels,3)
    print("Yon will probably like this person: ",
    resultList[classifierResult - 1])

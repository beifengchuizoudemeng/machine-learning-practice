from numpy import *

#SMO算法中的辅助函数
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append( [ float(lineArr[0]), float(lineArr[1]) ] )
        labelMat.append(float(lineArr[2]))
    fr.close()
    return dataMat, labelMat

#i是第一个alpha的下标，m是alpha的数目
#只要函数值不等于输入值i，函数就会进行随机选择
def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

#用于调整大于H或者小于L的alpha值
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

#简化版SMO
#dataMatIn数据集，classLabels类别标签，C与松弛变量有关的常数，可以通过对其的调节得到不同结果
#toler容错率， maxIter退出前最大循环次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):

    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)#m属性数目，n训练样本数目
    alphas = mat(zeros((m,1)))#列向量
    iter = 0

    while(iter < maxIter):

        alphaPairsChanged = 0 #用于记录alpha是否已经进行优化

        for i in range(m):
            #fXi预测值 Ei误差
            fXi = float(multiply(alphas, labelMat).T *\
                  (dataMatrix * dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])

            #如果误差很大，并且alpha不在边界上，就进入优化过程
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or\
               ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):

               j = selectJrand(i, m)
               fXj = float(multiply(alphas, labelMat).T * \
                     (dataMatrix * dataMatrix[j, :].T)) + b
               Ej = fXj - float(labelMat[j])

               alphaIold = alphas[i].copy()
               alphaJold = alphas[j].copy()

               if (labelMat[i] != labelMat[j]):
                   L = max(0, alphas[j] - alphas[i])
                   H = min(C, C + alphas[j] - alphas[i])
               else:
                   L = max(0, alphas[j] + alphas[i] - C)
                   H = min(C, alphas[j] + alphas[i])

               if L == H:
                   print("L == H")
                   continue

               eta = 2.0 *dataMatrix[i, :] * dataMatrix[j, :].T -\
                     dataMatrix[i, :] * dataMatrix[i, :].T -\
                     dataMatrix[j, :] * dataMatrix[j, :].T

               if eta >= 0:
                   print("eta >= 0")
                   continue

               alphas[j] -= labelMat[j] * (Ei - Ej) / eta
               alphas[j] = clipAlpha(alphas[j], H ,L)

               if (abs(alphas[j] - alphaJold) < 0.00001):
                   print("j not moving enough")
                   continue

               alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
               b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) *\
                    dataMatrix[i, :] * dataMatrix[i, :].T -\
                    labelMat[j] * (alphas[j] - alphaJold) *\
                    dataMatrix[i, :] * dataMatrix[j, :].T

               b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) *\
                    dataMatrix[i, :] * dataMatrix[j, :].T -\
                    labelMat[j] * (alphas[j] - alphaJold) *\
                    dataMatrix[j, :] * dataMatrix[j, :].T

               if (0 < alphas[i]) and (C > alphas[i]):
                   b = b1
               elif (0 < alphas[j]) and (C > alphas[j]):
                   b = b2
               else:
                   b = (b1 + b2) / 2.0

               alphaPairsChanged += 1
               print("iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))

        if (alphaPairsChanged == 0):
            iter += 1
        else : iter = 0
        print("iteration number: %d" % iter)
    return b, alphas

class optStruct:
     def __init__ (self, dataMatIn, classLabels, C, toler):
         self.X = dataMatIn
         self.labelMat = classLabels
         self.C = C
         self.tol = toler
         self.m = shape(dataMatIn)[0]
         self.alphas = mat(zeros((self.m, 1)))
         self.b = 0
          #第一列是eCache是否有效的标志位，第二列是实际的E值
         self.eCache = mat(zeros((self.m, 2)))#误差缓存

def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T *\
          (oS.X * oS.X[k,:].T)) + oS.b #f(xk)k是下标，预测值
    Ek = fXk - float(oS.labelMat[k])
    return Ek

#挑选第二个alpha，内循环中的启发式方法
def selectJ (i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei -Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

#完整Platt SMO算法中的优化例程
def innerL(i, oS):

    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or\
       ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):

        j, Ej = selectJ(i, oS, Ei) #第二个alpha选择中的启发式方法
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])

        if L==H:
             print ("L==H")
             return 0

        eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:] * oS.X[i,:].T - \
              oS.X[j,:] * oS.X[j,:].T

        if eta >= 0:
            print("eta>=0")
            return 0

        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j) #added this for the Ecache

        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print ("j not moving enough")
            return 0

        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] *\
                        (alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) *\
             oS.X[i,:] * oS.X[i,:].T - oS.labelMat[j] *\
             (oS.alphas[j] - alphaJold) * oS.X[i,:] * oS.X[j,:].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) *\
             oS.X[i,:] * oS.X[j,:].T - oS.labelMat[j] *\
             (oS.alphas[j] - alphaJold) * oS.X[j,:] * oS.X[j,:].T

        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1

    else:
        return 0

#完整版Platt SMO的外循环代码
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter:%d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas

def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w

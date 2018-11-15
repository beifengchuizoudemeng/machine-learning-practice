import logRegres
from numpy import*
dataArr, labelMat = logRegres.loadDataSet()
weights1 = logRegres.gradAscent(dataArr, labelMat)
#print(weights1)
logRegres.plotBestFit(weights1.getA())
weights2 = logRegres.stocGradAscent(array(dataArr), labelMat)
logRegres.plotBestFit(weights2)
weights3 = logRegres.stocGradAscentG(array(dataArr), labelMat)
logRegres.plotBestFit(weights3)

import trees
import treePlotter
myData,labels =trees.createDataSet()
#print(myData)
#print(labels)
#print(trees.calcShannonEnt(myData))
#retDataSet = trees.splitDataSet(myData,1,0)
#print(retDataSet)
#print(trees.chooseBestFeatureToSplit(myData))
myTree = trees.createTree(myData,labels)
print(myTree)
treePlotter.createPlot()

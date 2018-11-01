import trees
myDat,labels =trees.createDataSet()
print(myData)
print(labels)
print(trees.calcShannonEnt(myData))
#trees.splitDataSet(myData,0,0)
print(trees.chooseBestFeatureToSplit(myData))
mytree = trees.createTree(myDat,labels)
print(myTree)

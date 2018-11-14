import kNN
group,labels = kNN.createDataSet()
print(group)
print(labels)
result = kNN.classify0([0,0], group, labels, 3)
print(result)

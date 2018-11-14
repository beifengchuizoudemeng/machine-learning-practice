import bayes
listOPosts, listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)
print(myVocabList)
trainMat = []
for postinDocs in listOPosts:
    trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDocs))
print(trainMat)
p0V,p1V,pAb = bayes.trainNBO(trainMat, listClasses)
print(p0V, p1V,pAb)
bayes.testingNB()

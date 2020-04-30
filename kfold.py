import nltk # needed for Naive-Bayes
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold

# data is an array with our already pre-processed dataset examples
iris = load_iris()
feature_set = [{iris.feature_names[i] : d[i] for i in range(len(d))} for d in iris.data]
data = list(zip(feature_set, iris.target))

kf = KFold(n_splits=3)
sum = 0
for train, test in kf.split(data):
    train_data = np.array(data)[train]
    test_data = np.array(data)[test]
    classifier = nltk.NaiveBayesClassifier.train(train_data)
    classifier.show_most_informative_features()
    sum += nltk.classify.accuracy(classifier, test_data)
average = sum/3
print(average*100)

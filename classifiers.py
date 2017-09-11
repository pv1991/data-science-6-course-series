#!/usr/bin/python
# Usage: %run ./test.py - For running on IPYTHON
# ./test.py -<parameter>
# Parameters:
#			-b : Bernoulli Naives Bayes
#			-g : Gaussian Naives Bayes
#           -l : Linear Support Vector Classifier
#			-m : Multinomial Naives Bayes
#			-r : Random Forest
#			-s : Support Vector Classifier - RBF Kernel

# Author: Peter Vuong
# Date: 08/02/2016
# Last Modified: 29/02/2016

###############################################################################################
#                                 READ IN TRAINING/TEST DATA                                  #
###############################################################################################
	
# List of articles and categories
articles = list()
categories = list()
articlesTest = list()
categoriesTest = list()

# Read in training data for articles and add each article to a list.
f = open("data_train.txt")
for article in f.readlines():
	articles.append(article)
f.close()

# Read in test data for articles and add each article to a list.
f = open("data_valid.txt")
for article in f.readlines():
	articlesTest.append(article)
f.close()

# Read in training data for categories for each article and add each to a list.
f = open("labels_train_original.txt")
for category in f.readlines():
	categories.append(category)
f.close()

# Read in test data for categories for each article and add each to a list.
f = open("labels_valid_original.txt")
for category in f.readlines():
	categoriesTest.append(category)
f.close()
	
# Size of the articles and categories list
numArticles = len(articles)
numCategories = len(categories)

###############################################################################################
#                              BAG OF WORDS                                                   #
###############################################################################################

# Tokenizing text with scikit-learn
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(analyzer = 'word', 
                             tokenizer = None, 
							 preprocessor = None, 
							 stop_words = 'english',
							 max_features = 5000)
							 
# Fits the model and learns vocabulary. Transforms training data into feature
# vectors. Input: list of strings.
train_data_features = count_vect.fit_transform(articles)
train_data_features = train_data_features.toarray()

# Transforms training data into feature vectors. Input: list of strings.
test_data_features = count_vect.transform(articlesTest)
test_data_features = test_data_features.toarray()

###############################################################################################
#                               METHODS                                                       #
###############################################################################################                             

# Calculate the accuracy of the predictor.
def checkAccuracy(list1, list2):
	correct = 0;
	incorrect = 0;
	for i in range(len(list1)):
		#print("List1: " + list1[i].strip())
		#print("List2: " + list2[i].strip())
		
		if (list1[i].strip() == list2[i].strip()):
			correct += 1;
		else:
			incorrect += 1;
	
	total = correct + incorrect
	
	print("Number of correct predictions: " + str(correct))
	print("Number of incorrect predictions: " + str(incorrect))
	print("Total predictions: " + str(total))
	print("Predictor Accuracy: " + str(correct / total * 100) + "%")

# Print vocabulary words and number of times it appears.
def printVocabulary(trainFeatures, countVect):
	# Words in vocabulary.
	vocab = countVect.get_feature_names()

	import numpy as np
	dist = np.sum(trainFeatures, axis=0)
	for tag, count in zip(vocab, dist):
		print(count, tag)
	
# Count number of words of an item in given list.
def wordCount(list):
	for item in list:
		count = len(list.split())
	return count

###############################################################################################	
#                                   CLASSIFIERS	                                              #
###############################################################################################

#
# Bernoulli Naives Bayes classifier.
# Fit the classifer to the training set using bag of words
# as features and categories as response variable.
# Returns: accuracy of the predictor.	
def bernoulliNB(trainFeatures, testFeatures, responseVariable, testResult):
	from sklearn.naive_bayes import BernoulliNB
	bernoulli = BernoulliNB()
	bernoulli.fit(trainFeatures, responseVariable)
	BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
	
	# Use Bernoulli Naives Bayes to make prediction.
	result = bernoulli.predict(testFeatures)
	
	checkAccuracy(testResult, result)	
	
#
# Gaussian Naives Bayes classifier.
# Fit the classifer to the training set using bag of words
# as features and categories as response variable.
# Returns: accuracy of the predictor.	
def gaussianNB(trainFeatures, testFeatures, responseVariable, testResult):
	from sklearn.naive_bayes import GaussianNB
	gaussian = GaussianNB()
	gaussian.fit(trainFeatures, responseVariable)
	
	# Use Gaussian Naives Bayes to make prediction.
	result = gaussian.predict(testFeatures)
	
	checkAccuracy(testResult, result)		
	
#
# Linear Support Vector Classifier - Support Vector Machine.
# Fit the LinearSVC to the training set using bag of words
# as features and categories as response variable.
# Returns: accuracy of the predictor.
def linearSVC(trainFeatures, testFeatures, responseVariable, testResult):
	from sklearn.svm import LinearSVC
	svc = LinearSVC()
	svc = svc.fit(trainFeatures, responseVariable)
	LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
    intercept_scaling=1, loss='squared_hinge', max_iter=1000,
    multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
    verbose=0)
	
	# Use Linear Support Vector Classifer to make prediction.
	result = svc.predict(testFeatures)

	checkAccuracy(testResult, result)
	
#
# Multinomial Naives Bayes classifier.
# Fit the classifer to the training set using bag of words
# as features and categories as response variable.
# Returns: accuracy of the predictor.	
def multinomialNB(trainFeatures, testFeatures, responseVariable, testResult):
	from sklearn.naive_bayes import MultinomialNB
	multinomial = MultinomialNB()
	multinomial.fit(trainFeatures, responseVariable)
	MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
	
	# Use Multinomial Naives Bayes to make prediction.
	result = multinomial.predict(testFeatures)
	
	#checkAccuracy(testResult, result)
	multinomial.score(testRestul, result, sample_weight=None)
	
#
# Random Forest classifer with n trees.
# Fit the forest to the training set using bag of words
# as features and categories as response variable.
# Returns: accuracy of the predictor.
def randomForest(trainFeatures, testFeatures, responseVariable, nTrees, testResult):
	from sklearn.ensemble import RandomForestClassifier
	forest = RandomForestClassifier(n_estimators = nTrees)
	forest = forest.fit(trainFeatures, responseVariable)

	# Use Random Forest to make prediction.
	result = forest.predict(testFeatures)
	
	print(str(nTrees) + " trees\n")

	checkAccuracy(testResult, result)
	
#
# Support Vector Classifer - RBF Kernel Support Vector Machine.
# Fit the SVC to the training set using bag of words
# as features and categories as response variable.
# Returns: accuracy of the predictor.
def svcRBF(trainFeatures, testFeatures, responseVariable, testResult):
	from sklearn.svm import SVC
	svc = SVC()
	svc = svc.fit(trainFeatures, responseVariable)
	SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
	
	# Use Support Vector Classifer to make prediction.
	result = svc.predict(testFeatures)

	checkAccuracy(testResult, result)
	
###############################################################################################

import sys

if sys.argv[1] == "-b":
	print("Predictor: Bernoulli Naive Bayes")
	bernoulliNB(train_data_features, test_data_features, categories, categoriesTest)

elif sys.argv[1] == "-g":
	print("Predictor: Gaussian Naive Bayes")
	gaussianNB(train_data_features, test_data_features, categories, categoriesTest)
	
elif sys.argv[1] == "-l":
	print("Predictor: Linear Support Vector Classifer")
	linearSVC(train_data_features, test_data_features, categories, categoriesTest)
	
elif sys.argv[1] == "-m":
	print("Predictor: Multinomial Naive Bayes")
	multinomialNB(train_data_features, test_data_features, categories, categoriesTest)
	
elif sys.argv[1] == "-r":
	print("Predictor: Random Forest")
	randomForest(train_data_features, test_data_features, categories, 200, categoriesTest)

elif sys.argv[1] == "-s":
	print("Predictor: Support Vector Classifer - RBF Kernel")
	svcRBF(train_data_features, test_data_features, categories, categoriesTest)
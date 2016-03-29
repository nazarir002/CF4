# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 20:58:05 2016

@author: Daan Bouws
"""

import pickle

feature_set = pickle.load(open( "labeleddata0-10000.p", "rb" ))

train_set, test_set = feature_set[500:],feature_set[:500]

import nltk           
            
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
	    all_words.extend(words)
    return all_words
     
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features
    
word_features = get_word_features(get_words_in_tweets(train_set))
    
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features
    
train_set2 = nltk.classify.apply_features(extract_features, train_set)

classifier = nltk.NaiveBayesClassifier.train(train_set2)

#test_set = test_set[250:280]
print(2)
correctPrediction = 0
results = []
for complaint in test_set:
    testComplaint, category = complaint
    predictCategory = str(classifier.classify(extract_features(testComplaint)))
    if (predictCategory == category):
        correctPrediction+=1
    else:
        print(complaint)
    results.append(predictCategory+','+str(category))   

print('Result = ' + str(correctPrediction) +'/' +str(len(test_set))+'\t'+str((correctPrediction/len(test_set))*100)+'%')

file = open( "CurveClassifier.p", "wb" )
pickle.dump(classifier, file)
file.close()
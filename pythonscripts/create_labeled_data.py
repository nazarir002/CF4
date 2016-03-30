# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 12:26:32 2016

@author: Daan Bouws
"""
import pandas as pd

#https://oege.ie.hva.nl/phpmyadmin/ om data te doen
data = pd.read_csv('C:\\Users\\Richard\\Documents\\GitHub\\CF4\\chatlogs\\chatgood10k-20k.csv',sep=",",engine="python")
data2 = pd.read_csv('C:\\Users\\Richard\\Documents\\GitHub\\CF4\\chatlogs\\chat2.csv',sep=";",engine="python")
#
data.columns = ['A','B','C','D','chat','F','G']
chat = data['chat'].str.lower().str.split().tolist()
data2.columns = ['A','B','C','D','E','chat','G','H']
chat2 = data2['chat'].str.lower().str.split().tolist()

import string

def cleanseword(word):
    returnword = []
    for char in word:
        if char not in string.punctuation:
            returnword.append(char)
    return ''.join(returnword)

def cleansezin(zin):
    returnzin = []
    for word in zin:
        word2 = cleanseword(word)
        returnzin.append(word2)
    return returnzin


def remove_punctuation(data):
    data_nopunctuation = []
    for zin in data:
        if isinstance(zin,float)==False:
            zin2 = cleansezin(zin)
            data_nopunctuation.append(zin2)
    return data_nopunctuation
  
chat2 = remove_punctuation(chat2)     
chat = remove_punctuation(chat)

bad_words_english = pd.read_csv('C:\\Users\\Richard\\Documents\\GitHub\\CF4\\badwordslists\\bad_words_english.txt',sep="/n",engine='python')
bad_words = bad_words_english['label'].tolist()
bad_words_dutch = pd.read_csv('C:\\Users\\Richard\\Documents\\GitHub\\CF4\\badwordslists\\bad_words_dutch.txt',sep="/n",engine='python')
bad_words.append("idiot")
bad_words_french = pd.read_csv('C:\\Users\\Richard\\Documents\\GitHub\\CF4\\badwordslists\\bad_words_french.txt',sep='/n',engine='python')
bad_words = bad_words + bad_words_french['label'].tolist() + bad_words_dutch['label'].tolist()

def labelword(zin):
    string = "POSITIVE"
    for word in zin:
        if word in bad_words:
            string = "NEGATIVE"
    return string

def labelwords(data):
    allbad = []
    for zin in data:
        if ''.join(zin) in bad_words:
            allbad.append("NEGATIVE")
        elif ' '.join(zin) in bad_words:
            allbad.append("NEGATIVE")
        else:
            zin2 = labelword(zin)
            allbad.append(zin2)
    return allbad
    
chatlabels = labelwords(chat)
chat2labels = labelwords(chat2)

valueseries = pd.Series(chat)
df_chat = valueseries.to_frame()
df_chat.columns = ['Value']
df_chat['Label'] = pd.Series(chatlabels, index= df_chat.index)

feature = df_chat.Value

label = df_chat.Label

featuresets = [(label, feature)for index, (label, feature) in df_chat.iterrows()]

neg = 0
for item in chatlabels:
    if item=="NEGATIVE":
        neg+=1
    
print(str(neg)+" out of: "+str(len(chatlabels))+" lines contain foul language")

import pickle

file = open( "labeleddata.p", "wb" )
pickle.dump([featuresets], file)
file.close()

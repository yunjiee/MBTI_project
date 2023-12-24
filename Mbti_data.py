#import tweepy
import sys
import re
import string
import pickle
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
#from sklearn.externals import joblib

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
#from sklearn.svm import SVCsmatizermetext
#import CountVectorizer, TfidfTransformerusion_matrix, accuracy_score, classification_report, recall_score, cross_val_scorebliMultinomialNBdomForestClassifier, ExtraTreesClassifiersmatizermetext 

data = np.readline(".\data\archive\mbti_1.csv")

'''
#https://www.kaggle.com/code/rantan/multiclass-and-multi-output-classification#Text-Analysis-with-(MBTI)-Myers-Briggs-Personality-Type-Dataset
# 计算具有类型的主题列表 | 评论列表
from nltk.stem import PorterStemmer, WordNetLemmatizer

# 词形还原
stemmer = PorterStemmer()  # 词干提取器
lemmatiser = WordNetLemmatizer()  # 词形还原器
def pre_process_data(data, remove_stop_words=True):

    list_personality = []
    list_posts = []
    len_data = len(data)
    i=0
    
    for row in data.iterrows():
        i+=1
        if i % 500 == 0:
            print("%s | %s rows" % (i, len_data))

        ##### Remove and clean comments
        posts = row[1].posts
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', posts)
        temp = re.sub("[^a-zA-Z]", " ", temp)
        temp = re.sub(' +', ' ', temp).lower()
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])

        type_labelized = lab_encoder.transform([row[1].type])[0]
        list_personality.append(type_labelized)
        list_posts.append(temp)

    #del data
    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality

list_posts, list_personality = pre_process_data(data, remove_stop_words=True)           


'''

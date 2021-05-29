import sklearn
import pandas as pd
import matplotlib as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import re

posReviews = []
negReviews = []
reviews = []

def readReviews(path1, list):
    for filename in os.listdir(path1):
        f = open(os.path.join(path1, filename), "r")
        list.append(f.read())

def preprocessing(list):
    for i in range(len(list)):
        list[i] = " ".join([word for word in word_tokenize(list[i]) if word not in stopwords.words()])
        list[i] = re.sub('[\W]+', ' ', list[i])

def generateTF_IDF():
    tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(use_idf=True)
    tfidf_vectors = tfidf_vectorizer.fit_transform(reviews)


# Press the green button in the gutter to run the script.
path1 = "E:\\FCAI\\4th year , 2nd semester\\NLP\\Assignments\\Assignment 2\\review_polarity\\txt_sentoken\\pos"
path2 = "E:\\FCAI\\4th year , 2nd semester\\NLP\\Assignments\\Assignment 2\\review_polarity\\txt_sentoken\\neg"
readReviews(path1, posReviews)
readReviews(path2, negReviews)
reviews = posReviews
reviews.extend(negReviews)
preprocessing(reviews)
print(reviews[0])

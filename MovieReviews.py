import pandas as pd
import matplotlib as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import os
import re

reviews = []

def readReviews(path1):
    for filename in os.listdir(path1):
        f = open(os.path.join(path1, filename), "r")
        reviews.append(f.read())

def preprocessing():
    for i in range(len(reviews)):
        reviews[i] = re.sub('[\W]+', ' ', reviews[i])
        tokens = word_tokenize(reviews[i])
        for word in tokens:
            if word in stopwords.words():
                reviews[i] = reviews[i].replace(" "+word+" ", " ")
        break

def generateTF_IDF():
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_vectors = tfidf_vectorizer.fit_transform([reviews[0]])
    df = pd.DataFrame(tfidf_vectors[0].T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
    df.sort_values(by=["tfidf"], ascending=False)
    print(df)


path1 = "E:\\FCAI\\4th year , 2nd semester\\NLP\\Assignments\\Assignment 2\\review_polarity\\txt_sentoken\\pos"
path2 = "E:\\FCAI\\4th year , 2nd semester\\NLP\\Assignments\\Assignment 2\\review_polarity\\txt_sentoken\\neg"
readReviews(path1)
readReviews(path2)
preprocessing()
print(reviews[0])
print()
generateTF_IDF()

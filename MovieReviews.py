import pandas as pd
import matplotlib as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import os
import re

from sklearn.model_selection import train_test_split

reviews = []
reviewLabel = []

def divideData(tfidf_vec):
    X_train, X_test, y_train, y_test = train_test_split(tfidf_vec, reviewLabel, test_size=0.2)
    print(X_test)


def readReviews(path):
    counter = 0
    for filename in os.listdir(path):
        if(counter == 6):
            break
        counter=counter+1
        f = open(os.path.join(path, filename), "r")
        reviews.append(f.read())

def preprocessing():
    for i in range(len(reviews)):
        reviews[i] = re.sub('[\W]+', ' ', reviews[i])
        tokens = word_tokenize(reviews[i])
        for word in tokens:
            if word in stopwords.words():
                reviews[i] = reviews[i].replace(" "+word+" ", " ")
    reviewLabel = [1] * 1000
    reviewLabel.extend([0] * 1000)

def generateTF_IDF():
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_vectors = tfidf_vectorizer.fit_transform(reviews)
    df = pd.DataFrame(tfidf_vectors[0].T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
    df.sort_values(by=["tfidf"], ascending=False)
    print(df)
    return tfidf_vectors



def main():
    path1 = "E:\\FCI\\fourth year\\Second term\\NLP\\Assignments\\Assignment 2\\txt_sentoken\\pos"
    path2 = "E:\\FCI\\fourth year\\Second term\\NLP\\Assignments\\Assignment 2\\txt_sentoken\\neg"
    readReviews(path1)
    readReviews(path2)
    preprocessing()
    print(reviews[0])
    print()
    tfidf_vectors = generateTF_IDF()
    print()
    divideData(tfidf_vectors)

main()

#clf = linear_model.SGDClassifier()
#clf.fit(tfidf_vec, reviewLabel)
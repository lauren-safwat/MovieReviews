import pandas as pd
import numpy as np
import matplotlib as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import os
import re

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import LinearSVC

reviews = []
reviewLabel = np.zeros(2000)


def trainTF_IDF():
    kf = StratifiedKFold(n_splits=5, random_state=2, shuffle=True)
    totalsvm = 0  # Accuracy measure on 2000 files
    totalMatSvm = np.zeros((2, 2))  # Confusion matrix on 2000 files

    for train_index, test_index in kf.split(reviews, reviewLabel):
        X_train = [reviews[i] for i in train_index]
        X_test = [reviews[i] for i in test_index]
        y_train, y_test = reviewLabel[train_index], reviewLabel[test_index]

        vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True, stop_words='english')
        train_corpus_tf_idf = vectorizer.fit_transform(X_train)
        test_corpus_tf_idf = vectorizer.transform(X_test)
        model = LinearSVC()
        model.fit(train_corpus_tf_idf, y_train)
        result = model.predict(test_corpus_tf_idf)

        totalMatSvm = totalMatSvm + confusion_matrix(y_test, result)
        totalsvm = totalsvm + sum(y_test == result)

    return totalMatSvm, totalsvm


def readReviews(path):
    for filename in os.listdir(path):
        f = open(os.path.join(path, filename), "r")
        reviews.append(f.read())


def preprocessing():
    reviewLabel[0:1000] = 1
    for i in range(len(reviews)):
        reviews[i] = reviews[i].lower()
        reviews[i] = re.sub('[^\sa-zA-Z]', '', reviews[i])
        reviews[i] = " ".join([word for word in word_tokenize(reviews[i]) if word not in stopwords.words()])


def generateTF_IDF():
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_vectors = tfidf_vectorizer.fit_transform(reviews)
    df = pd.DataFrame(tfidf_vectors[0].T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
    df.sort_values(by=["tfidf"], ascending=False)
    print(df)
    return tfidf_vectors.toarray()


def main():
    path1 = "..\\txt_sentoken\\pos"
    path2 = "..\\txt_sentoken\\neg"
    readReviews(path1)
    readReviews(path2)
    preprocessing()
    totalMatSvm, totalsvm = trainTF_IDF()
    print("totalMatSvm: ", totalMatSvm)
    print("totalsvm: ", totalsvm/2000)

main()

#clf = linear_model.SGDClassifier()
#clf.fit(tfidf_vec, reviewLabel)
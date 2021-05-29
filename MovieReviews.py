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
vectorizer = TfidfVectorizer(use_idf=True)

def trainTF_IDF():
    kf = StratifiedKFold(n_splits=5, random_state=3, shuffle=True)
    totalsvm = 0
    totalMatSvm = np.zeros((2, 2))

    for train_index, test_index in kf.split(reviews, reviewLabel):
        X_train = [reviews[i] for i in train_index]
        X_test = [reviews[i] for i in test_index]
        y_train, y_test = reviewLabel[train_index], reviewLabel[test_index]

        train_corpus_tf_idf = vectorizer.fit_transform(X_train)
        test_corpus_tf_idf = vectorizer.transform(X_test)
        model = LinearSVC()
        model.fit(train_corpus_tf_idf, y_train)
        result = model.predict(test_corpus_tf_idf)

        totalMatSvm = totalMatSvm + confusion_matrix(y_test, result)
        totalsvm = totalsvm + sum(y_test == result)

    return model, totalMatSvm, totalsvm


def readReviews(path):
    for filename in os.listdir(path):
        f = open(os.path.join(path, filename), "r")
        reviews.append(f.read())


def preprocessing():
    reviewLabel[0:1000] = 1
    cachedStopWords = stopwords.words("english")
    for i in range(len(reviews)):
        reviews[i] = reviews[i].lower()
        reviews[i] = re.sub("[^\sa-zA-Z]", "", reviews[i])
        reviews[i] = ' '.join([word for word in word_tokenize(reviews[i]) if word not in cachedStopWords])


def testModel(model):
    while(True):
        print("Enter your movie review: ")
        lines = []
        while True:
            line = input()
            if line:
                lines.append(line)
            else:
                break
        review = '\n'.join(lines)

        tfidf = vectorizer.transform([review])
        yPredicted = model.predict(tfidf)
        if(yPredicted == 0):
            print("Negative Review")
        else:
            print("Positive Review")


def main():
    path1 = "..\\review_polarity\\txt_sentoken\\pos"
    path2 = "..\\review_polarity\\txt_sentoken\\neg"
    readReviews(path1)
    readReviews(path2)
    preprocessing()
    model, totalMatSvm, totalsvm = trainTF_IDF()
    print("totalMatSvm: ", totalMatSvm)
    print("totalsvm: ", (totalsvm/2000)*100, "%")
    print()
    testModel(model)

main()
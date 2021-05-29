import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from yellowbrick.text import TSNEVisualizer
import os
import re

# ============================================================================

reviews = []
reviewsLabel = np.zeros(2000)
tfidf_vectorizer = TfidfVectorizer(use_idf=True)

# ============================================================================


def readReviews(path):
    for filename in os.listdir(path):
        f = open(os.path.join(path, filename), "r")
        reviews.append(f.read())


# ============================================================================


def preprocessing():
    reviewsLabel[0:1000] = 1
    cachedStopWords = stopwords.words("english")
    for i in range(len(reviews)):
        reviews[i] = reviews[i].lower()
        reviews[i] = re.sub("[^\sa-zA-Z]", "", reviews[i])
        reviews[i] = ' '.join([word for word in word_tokenize(reviews[i]) if word not in cachedStopWords])


# ============================================================================


def trainTF_IDF():
    kf = StratifiedKFold(n_splits=5, random_state=3, shuffle=True)
    nCorrectPred = 0

    for train_index, test_index in kf.split(reviews, reviewsLabel):
        X_train = [reviews[i] for i in train_index]
        X_test = [reviews[i] for i in test_index]
        Y_train, Y_test = reviewsLabel[train_index], reviewsLabel[test_index]

        trainTF_IDF = tfidf_vectorizer.fit_transform(X_train)
        testTF_IDF = tfidf_vectorizer.transform(X_test)
        model = LinearSVC()
        model.fit(trainTF_IDF, Y_train)
        result = model.predict(testTF_IDF)

        nCorrectPred = nCorrectPred + sum(Y_test == result)

    return model, nCorrectPred


# ============================================================================


def testModel(model):
    print("Enter your movie review: ")
    lines = []
    while True:
        line = input()
        if line:
            lines.append(line)
        else:
            break
    review = '\n'.join(lines)

    tfidf = tfidf_vectorizer.transform([review])
    yPredicted = model.predict(tfidf)
    if (yPredicted == 0):
        print("Negative Review")
    else:
        print("Positive Review")


# ============================================================================


def plotData():
    tfidf = TfidfVectorizer()
    docs = tfidf.fit_transform(reviews)
    labels = reviewsLabel

    tsne = TSNEVisualizer()
    tsne.fit_transform(docs, labels)
    tsne.poof()


# ============================================================================


def main():
    path1 = "..\\review_polarity\\txt_sentoken\\pos"
    path2 = "..\\review_polarity\\txt_sentoken\\neg"
    readReviews(path1)
    readReviews(path2)
    preprocessing()
    model, nCorrectPred = trainTF_IDF()
    print("Accuracy: ", (nCorrectPred / 2000) * 100, "%")
    print()
    testModel(model)
    plotData()

main()

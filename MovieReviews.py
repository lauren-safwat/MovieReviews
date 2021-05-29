import numpy as np
import matplotlib as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC


reviews = []
reviewLabel = np.zeros(2000)
vectorizer = TfidfVectorizer(use_idf=True)

def trainTF_IDF():
    kf = StratifiedKFold(n_splits=5, random_state=3, shuffle=True)
    nCorrectpred = 0

    for train_index, test_index in kf.split(reviews, reviewLabel):
        X_train = [reviews[i] for i in train_index]
        X_test = [reviews[i] for i in test_index]
        Y_train, Y_test = reviewLabel[train_index], reviewLabel[test_index]

        trainTF_IDF = vectorizer.fit_transform(X_train)
        testTF_IDF = vectorizer.transform(X_test)
        model = LinearSVC()
        model.fit(trainTF_IDF, Y_train)
        result = model.predict(testTF_IDF)

        nCorrectpred = nCorrectpred + sum(Y_test == result)

    return model, nCorrectpred


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
    while True:
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


#-----------------------------------------------------------------------------

def plotData( X, feature2):
    fig, ax = plt.subplots(figsize=(5, 5))
    positive = data[data['target'].isin([1])]
    negative = data[data['target'].isin([0])]

    ax.scatter(positive[feature1], positive[feature2], s=50, c='g', marker='x', label='positive review')
    ax.scatter(negative[feature1], negative[feature2], s=50, c='r', marker='_', label='negative review')
    ax.legend()
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    plotLine(X, w, b)



def main():
    path1 = "..\\txt_sentoken\\pos"
    path2 = "..\\txt_sentoken\\neg"
    readReviews(path1)
    readReviews(path2)
    preprocessing()
    model, nCorrectPred = trainTF_IDF()
    print("Accuracy: ", (nCorrectPred/2000)*100, "%")
    print()
    testModel(model)

main()
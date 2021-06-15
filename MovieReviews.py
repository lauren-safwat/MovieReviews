import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from yellowbrick.text import TSNEVisualizer
import os
import re
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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
    stopWords = stopwords.words("english")
    moreChars = ['\'s', '\'re', '(', '?', ')', '.', ',', ':', '-', '/', '``', '\'ll']
    stopWords.remove('not')
    stopWords.extend(moreChars)

    for i in range(len(reviews)):
        gensim.utils.simple_preprocess(reviews[i])
        reviews[i] = re.sub("[0-9]", "", reviews[i])
        reviews[i] = re.sub("[/]", ' ', reviews[i])
        reviews[i] = [word for word in word_tokenize(reviews[i]) if word not in stopWords and (len(word) > 2)]


# ============================================================================


def createWordEmbedding():
    model = gensim.models.Word2Vec(reviews, vector_size=150, window=5, min_count=1, workers=8)
    model.train(reviews, total_examples=len(reviews), epochs=10)
    return model


# ============================================================================


def sentenceEmbedding(model):
    reviewFeatureVec = []
    for review in reviews:
        review = [word for word in review if word in model.wv.index_to_key]    # remove out-of-vocabulary words
        if len(review) >= 1:
            reviewFeatureVec.append(np.mean(model.wv[review], axis=0))
        else:
            reviewFeatureVec.append([])

    return reviewFeatureVec

# ============================================================================


def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# ============================================================================




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


def testModel(model, is_tfidf):
    print("Enter your movie review: ")
    lines = []
    while True:
        line = input()
        if line:
            lines.append(line)
        else:
            break
    review = '\n'.join(lines)

    if is_tfidf:
        review = tfidf_vectorizer.transform([review])

    # else:
    #     review =

    yPredicted = model.predict(review)
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
    # model, nCorrectPred = trainTF_IDF()
    # print("Accuracy: ", (nCorrectPred / 2000) * 100, "%")
    # print()
    # testModel(model, true)
    #plotData()

    word2vecModel = createWordEmbedding()
    reviewFeatureVec = sentenceEmbedding(word2vecModel)



main()

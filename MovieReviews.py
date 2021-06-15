import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from yellowbrick.text import TSNEVisualizer
import os
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# ============================================================================

tfidf_vectorizer = TfidfVectorizer(use_idf=True)

# ============================================================================


def readReviews(path, reviews):
    for filename in os.listdir(path):
        f = open(os.path.join(path, filename), "r")
        reviews.append(f.read())

    return reviews


# ============================================================================


def preprocessing(reviews, reviewsLabel):
    reviewsLabel[0:1000] = 1
    stopWords = stopwords.words("english")
    stopWords.remove('not')

    for i in range(len(reviews)):
        reviews[i] = [word for word in word_tokenize(reviews[i]) if word not in stopWords and (len(word) > 2)]
        reviews[i] = [word for word in reviews[i] if word.isalpha() or word.find("n't") != -1]

    return reviews, reviewsLabel


# ============================================================================


def createWordEmbedding(reviews):
    model = gensim.models.Word2Vec(reviews, vector_size=1000, window=20, min_count=1, workers=15)
    model.train(reviews, total_examples=len(reviews), epochs=100)
    return model


# ============================================================================


def sentenceEmbedding(model, reviews):
    reviewFeatureVec = []
    for review in reviews:
        review = [word for word in review if word in model.wv.index_to_key]    # remove out-of-vocabulary words
        if len(review) >= 1:
            reviewFeatureVec.append(np.mean(model.wv[review], axis=0))
        else:
            reviewFeatureVec.append([])
    print(reviewFeatureVec[0])
    return reviewFeatureVec


# ============================================================================


def svcHyperparam(X, y):
    model = SVC()

    kernel = ['poly', 'rbf', 'sigmoid', 'linear']
    C = [100, 50, 10, 1.0, 0.1]

    grid = dict(kernel=kernel, C=C)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=3)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X, y)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f with: %r" % (mean, param))


# ============================================================================


def randForestHparam(X, y):
    model = RandomForestClassifier()

    n_estimators = [10, 100, 1000]
    max_features = ['sqrt', 'log2']

    grid = dict(n_estimators=n_estimators, max_features=max_features)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=3)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X, y)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f with: %r" % (mean, param))


# ============================================================================


def logisticHyperparam(X, y):
    model = LogisticRegression()

    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    c_values = [100, 10, 1.0, 0.1, 0.01]

    grid = dict(solver=solvers, C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=3)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X, y)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f with: %r" % (mean, param))


# ============================================================================


def trainTF_IDF_SVM(X, y):
    kf = StratifiedKFold(n_splits=5, random_state=3, shuffle=True)
    nCorrectPred = 0
    model = SVC(kernel='sigmoid', C=10, gamma=0.001)

    for train_index, test_index in kf.split(X, y):
        X_train = [' '.join(X[i]) for i in train_index]
        X_test = [' '.join(X[i]) for i in test_index]
        Y_train, Y_test = y[train_index], y[test_index]

        trainTF_IDF = tfidf_vectorizer.fit_transform(X_train)
        testTF_IDF = tfidf_vectorizer.transform(X_test)
        model.fit(trainTF_IDF, Y_train)
        result = model.predict(testTF_IDF)

        nCorrectPred = nCorrectPred + sum(Y_test == result)

    return model, nCorrectPred


# ============================================================================


def trainTF_IDF_LR(X, y):
    kf = StratifiedKFold(n_splits=5, random_state=3, shuffle=True)
    nCorrectPred = 0

    for train_index, test_index in kf.split(X, y):
        X_train = [' '.join(X[i]) for i in train_index]
        X_test = [' '.join(X[i]) for i in test_index]
        Y_train, Y_test = y[train_index], y[test_index]

        trainTF_IDF = tfidf_vectorizer.fit_transform(X_train)
        testTF_IDF = tfidf_vectorizer.transform(X_test)
        model = LogisticRegression(C=0.1, solver='newton-cg', penalty='l2')
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

    review = tfidf_vectorizer.transform([review])

    yPredicted = model.predict(review)
    if (yPredicted == 0):
        print("Negative Review")
    else:
        print("Positive Review")


# ============================================================================


def plotData(X, y):
    tfidf = TfidfVectorizer()
    docs = tfidf.fit_transform(X)
    labels = y

    tsne = TSNEVisualizer()
    tsne.fit_transform(docs, labels)
    tsne.poof()


# ============================================================================


def main():
    reviews = []
    reviewsLabel = np.zeros(2000)

    path1 = "..\\review_polarity\\txt_sentoken\\pos"
    path2 = "..\\review_polarity\\txt_sentoken\\neg"
    reviews = readReviews(path1, reviews)
    reviews = readReviews(path2, reviews)
    reviews, reviewsLabel = preprocessing(reviews, reviewsLabel)

    """ TF-IDF - SVM"""
    model, nCorrectPred = trainTF_IDF_SVM(reviews, reviewsLabel)
    print("Accuracy: ", (nCorrectPred / 2000) * 100, "%")
    print()
    testModel(model)
    plotData()

    """ TF-IDF - LR"""
    model, nCorrectPred = trainTF_IDF_LR(reviews, reviewsLabel)
    print("Accuracy: ", (nCorrectPred / 2000) * 100, "%")
    print()
    testModel(model)
    plotData()

    """ Word and Sentence Embedding Model """
    word2vecModel = createWordEmbedding(reviews)
    reviewFeatureVec = sentenceEmbedding(word2vecModel, reviews)

    """ Tuning hyperparameters for LR Classifier"""
    logisticHyperparam(reviewFeatureVec, reviewsLabel)

    """ Tuning hyperparameters for SVC Classifier"""
    svcHyperparam(reviewFeatureVec, reviewsLabel)

    """ Tuning hyperparameters for RF Classifier"""
    randForestHparam(reviewFeatureVec, reviewsLabel)


main()

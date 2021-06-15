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
    stopWords.remove('not')

    for i in range(len(reviews)):
        reviews[i] = [word for word in word_tokenize(reviews[i]) if word not in stopWords and (len(word) > 2)]
        reviews[i] = [word for word in reviews[i] if word.isalpha() or word.find("n't") != -1]

    print(' '.join(reviews[0]))


# ============================================================================


def createWordEmbedding():
    model = gensim.models.Word2Vec(reviews, vector_size=1200, window=20, min_count=1, workers=8)
    model.train(reviews, total_examples=len(reviews), epochs=100)
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


def svcHyperparam(X, y):
    model = SVC()
    kernel = ['poly', 'rbf', 'sigmoid', 'linear']
    C = [100, 50, 10, 1.0, 0.1]
    gamma = [0.001, 0.01, 0.1, 1, 10]
    # define grid search
    grid = dict(kernel=kernel, C=C, gamma=gamma)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=3)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X, y)
    # summarize results
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
    # define grid search
    grid = dict(n_estimators=n_estimators, max_features=max_features)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=3)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f with: %r" % (mean, param))


# ============================================================================


def logisticHyperparam(X, y):
    model = LogisticRegression()
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    # define grid search
    grid = dict(solver=solvers, penalty=penalty, C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=3)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


# ============================================================================


def trainSVM_Classifier(featureVec):
    kf = StratifiedKFold(n_splits=5, random_state=3, shuffle=True)
    nCorrectPred = 0
    model = LinearSVC()

    for train_index, test_index in kf.split(featureVec, reviewsLabel):
        X_train = [featureVec[i] for i in train_index]
        X_test = [featureVec[i] for i in test_index]
        Y_train, Y_test = reviewsLabel[train_index], reviewsLabel[test_index]

        model.fit(X_train, Y_train)
        result = model.predict(X_test)

        nCorrectPred += sum(Y_test == result)

    return model, nCorrectPred/2000


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

    """ TF-IDF """
    # model, nCorrectPred = trainTF_IDF()
    # print("Accuracy: ", (nCorrectPred / 2000) * 100, "%")
    # print()
    # testModel(model, true)
    #plotData()

    """ Word and Sentence Embedding Model """
    word2vecModel = createWordEmbedding()
    reviewFeatureVec = sentenceEmbedding(word2vecModel)
    logisticHyperparam(reviewFeatureVec, reviewsLabel)

    # svmModel, accuracy = trainSVM_Classifier(reviewFeatureVec)
    # print("Accuracy: ", accuracy * 100, "%")

    #svcHyperparam(reviewFeatureVec, reviewsLabel)
    #randForestHparam(reviewFeatureVec, reviewsLabel)


main()

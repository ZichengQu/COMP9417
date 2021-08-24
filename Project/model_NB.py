import numpy as np
import sklearn.metrics as ms
from sklearn import naive_bayes

import preprocessing

train_X, train_Y, test_X, test_Y, train, test = preprocessing.getDataSet(dataSet = 1) # shape: (59381, 89), (59381,), (19765, 89), None, (59381, 128), (19765, 127)

train_Y = np.array(train_Y).reshape(-1, 1) # shape: (59381, 1)

train_Y = [i for j in train_Y for i in j]
train_xset = train_X[:50000]
test_xset = train_X[50000:]
train_yset = train_Y[:50000]
test_yset = train_Y[50000:]

def showLog(name, best_alpha, train_acc, test_acc, recall, prec):
    print('The result of {}'.format(name))
    print('best alpha: ', best_alpha)
    print('train accuracy:', train_acc)
    print('test accuracy:', test_acc)
    print('recall:', recall)
    print('precision: ', prec)
    print('-------------------------')

def get_BernoulliNB_alpha(*data):
    X_train, X_test, y_train, y_test = data
    alphas = np.logspace(-2, 5, num=200)
    train_scores = []
    test_scores = []
    for alpha in alphas:
        cls = naive_bayes.BernoulliNB(alpha=alpha)
        cls.fit(X_train, y_train)
        train_scores.append(cls.score(X_train, y_train))
        test_scores.append(cls.score(X_test, y_test))
    alphas = alphas.tolist()
    best_alpha = alphas[test_scores.index(max(test_scores))]
    cls = naive_bayes.BernoulliNB(alpha=best_alpha)
    cls.fit(X_train, y_train)
    test_acc = cls.score(X_test, y_test)
    train_acc = cls.score(X_train, y_train)
    pred_y = cls.predict(X_test)
    pred_y = pred_y.tolist()
    recall = ms.recall_score(y_test, pred_y, average='macro')
    prec = ms.precision_score(y_test, pred_y, average='macro')
    showLog("BernoulliNB", best_alpha, train_acc, test_acc, recall, prec)

get_BernoulliNB_alpha(train_xset, test_xset, train_yset, test_yset)

def get_MultinomialNB_alpha(*data):
    X_train, X_test, y_train, y_test = data
    alphas = np.logspace(-2, 5, num=200)
    train_scores = []
    test_scores = []
    for alpha in alphas:
        cls = naive_bayes.MultinomialNB(alpha=alpha)
        cls.fit(X_train, y_train)
        train_scores.append(cls.score(X_train, y_train))
        test_scores.append(cls.score(X_test, y_test))
    alphas = alphas.tolist()
    best_alpha = alphas[test_scores.index(max(test_scores))]
    cls = naive_bayes.MultinomialNB(alpha=best_alpha)
    cls.fit(X_train, y_train)
    test_acc = cls.score(X_test, y_test)
    train_acc = cls.score(X_train, y_train)
    pred_y = cls.predict(X_test)
    pred_y = pred_y.tolist()
    recall = ms.recall_score(y_test, pred_y, average='macro')
    prec = ms.precision_score(y_test, pred_y, average='macro')
    showLog("MultinomialNB", best_alpha, train_acc, test_acc, recall, prec)

get_MultinomialNB_alpha(train_xset, test_xset, train_yset, test_yset)
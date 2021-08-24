import numpy as np
import sklearn.metrics as ms
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

import preprocessing

train_X, train_Y, test_X, test_Y, train, test = preprocessing.getDataSet(dataSet = 1) # shape: (59381, 89), (59381,), (19765, 89), None, (59381, 128), (19765, 127)

train_Y = np.array(train_Y).reshape(-1, 1) # shape: (59381, 1)

train_Y = [i for j in train_Y for i in j]
train_xset = train_X[:50000]
test_xset = train_X[50000:]
train_yset = train_Y[:50000]
test_yset = train_Y[50000:]


def showLog1(name, train_acc, test_acc, recall, prec):
    '''
    Show the printed log for Case 1.
    '''
    print('The result of Logistic Regression({})'.format(name))
    print('train accuracy:', train_acc)
    print('test accuracy:', test_acc)
    print('recall: ', recall)
    print('precision: ', prec)
    print('--------')

def showLog2_3(train_acc, test_acc, best_params_):
    '''
    Show the printed log for Case 2 and Case 3.
    '''
    print('train acc:',train_acc)
    print('test acc:',test_acc)
    print('best c is: ',best_params_)
    print('--------')

# Case 1. The results under the default parameters.
cls1 = LogisticRegression(penalty='l1', solver='liblinear')
cls1.fit(train_xset, train_yset)

test_acc1 = cls1.score(test_xset, test_yset)
train_acc1 = cls1.score(train_xset, train_yset)
pred_test1 = cls1.predict(test_xset)
pred_test1 = pred_test1.tolist()
recall1 = ms.recall_score(test_yset, pred_test1, average='macro')
prec1 = ms.precision_score(test_yset, pred_test1, average='macro')
showLog1("penalty=l1, solver=liblinear", train_acc1, test_acc1, recall1, prec1)

cls2 = LogisticRegression(penalty='l2', solver='liblinear')
cls2.fit(train_xset, train_yset)

test_acc2 = cls2.score(test_xset, test_yset)
train_acc2 = cls2.score(train_xset, train_yset)
pred_test2 = cls2.predict(test_xset)
pred_test2 = pred_test2.tolist()
recall2 = ms.recall_score(test_yset, pred_test2, average='macro')
prec2 = ms.precision_score(test_yset, pred_test2, average='macro')
showLog1("penalty=l2, solver=liblinear", train_acc2, test_acc2, recall2, prec2)

cls3 = LogisticRegression(penalty='l2', solver='saga', max_iter=10000)
cls3.fit(train_xset, train_yset)

test_acc3 = cls3.score(test_xset, test_yset)
train_acc3 = cls3.score(train_xset, train_yset)
pred_test3 = cls3.predict(test_xset)
pred_test3 = pred_test3.tolist()
recall3 = ms.recall_score(test_yset, pred_test3, average='macro')
prec3 = ms.precision_score(test_yset, pred_test3, average='macro')
showLog1("penalty=l2, solver=saga, max_iter=10000", train_acc3, test_acc3, recall3, prec3)

cls4 = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=10000)
cls4.fit(train_xset, train_yset)

test_acc4 = cls4.score(test_xset, test_yset)
train_acc4 = cls4.score(train_xset, train_yset)
pred_test4 = cls4.predict(test_xset)
pred_test4 = pred_test4.tolist()
recall4 = ms.recall_score(test_yset, pred_test4, average='macro')
prec4 = ms.precision_score(test_yset, pred_test4, average='macro')
showLog1("penalty=l2, solver=lbfgs, max_iter=10000", train_acc4, test_acc4, recall4, prec4)

cls5 = LogisticRegression(penalty='l2', solver='newton-cg', max_iter=10000)
cls5.fit(train_xset, train_yset)

test_acc5 = cls5.score(test_xset, test_yset)
train_acc5 = cls5.score(train_xset, train_yset)
pred_test5 = cls5.predict(test_xset)
pred_test5 = pred_test5.tolist()
recall5 = ms.recall_score(test_yset, pred_test5, average='macro')
prec5 = ms.precision_score(test_yset, pred_test5, average='macro')
showLog1("penalty=l2, solver=newton-cg, max_iter=10000", train_acc5, test_acc5, recall5, prec5)

# Case 2. Test logistic (penalty=l1、l2， solver = liblinear)
kf = KFold(n_splits=5)
def floatrange(start,stop,steps):
    return [round(start + float(i) * (stop - start) / (float(steps) - 1) , 4) for i in range(steps)]
c_grid1 = list(floatrange(0.0001, 1.0, 10))
c_grid2 = list(floatrange(1.0, 2.0, 10))
param_grid = { 'C':c_grid1} # Choose from c_grid1 or c_grid2

cls_logistic1 = LogisticRegression(penalty='l1', solver='liblinear')
cls = GridSearchCV(estimator=cls_logistic1,cv=5,param_grid=param_grid, verbose=10)
cls.fit(train_xset, train_yset)
test_acc = cls.score(test_xset, test_yset)
train_acc = cls.score(train_xset, train_yset)
showLog2_3(train_acc, test_acc, cls.best_params_)

cls_logistic2 = LogisticRegression(penalty='l2', solver='liblinear')
cls = GridSearchCV(estimator=cls_logistic2,cv=5,param_grid=param_grid, verbose=10)
cls.fit(train_xset, train_yset)
test_acc = cls.score(test_xset, test_yset)
train_acc = cls.score(train_xset, train_yset)
showLog2_3(train_acc, test_acc, cls.best_params_)

# Case 3. Test logistic (penalty=l2， solver = all range)
c_grid = list(floatrange(0.5, 1.5, 11))
param_grid = { 'C':c_grid}

# Five Situations：
# (penalty='l2', solver='liblinear')
# (penalty='l1', solver='liblinear')
# (penalty='l2', solver='saga', max_iter=3000)
# (penalty='l2', solver='lbfgs', max_iter=3000)
# (penalty='l2', solver='newton-cg', max_iter=3000)
# The last three take quite a long time to train.
# solver = 'liblinear' is suitable for l1 and l2，others are only suitable for l2.
cls_logistic1 = LogisticRegression(penalty='l2', solver='saga', max_iter=8000)
cls1 = GridSearchCV(estimator=cls_logistic1,cv=5,param_grid=param_grid, verbose=10)
cls1.fit(train_xset, train_yset)
test_acc1 = cls1.score(test_xset, test_yset)
train_acc1 = cls1.score(train_xset, train_yset)

cls_logistic2 = LogisticRegression(penalty='l2', solver='lbfgs',max_iter=5000)
cls2 = GridSearchCV(estimator=cls_logistic2,cv=5,param_grid=param_grid, verbose=10)
cls2.fit(train_xset, train_yset)
test_acc2 = cls2.score(test_xset, test_yset)
train_acc2 = cls2.score(train_xset, train_yset)

cls_logistic3 = LogisticRegression(penalty='l2', solver='newton-cg', max_iter=5000)
cls3 = GridSearchCV(estimator=cls_logistic3,cv=5,param_grid=param_grid, verbose=10)
cls3.fit(train_xset, train_yset)
test_acc3 = cls3.score(test_xset, test_yset)
train_acc3 = cls3.score(train_xset, train_yset)

showLog2_3(train_acc1, test_acc1, cls1.best_params_)
showLog2_3(train_acc2, test_acc2, cls2.best_params_)
showLog2_3(train_acc3, test_acc3, cls3.best_params_)
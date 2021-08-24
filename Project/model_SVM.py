import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

import preprocessing

# Get the data set from the preprocessing.py.
train_X, train_Y, test_X, test_Y, train, test = preprocessing.getDataSet(dataSet = 4) # shape: (59381, 89), (59381,), (19765, 89), None, (59381, 128), (19765, 127)

# 利用 train_test_split 拆分训练集
X_train, X_test, y_train, y_test = train_test_split(train_X, train.loc[:, "Response"], random_state=0, test_size=0.3)

parameters = {"kernel": ['rbf'],
                 "C": [1]
                 }

model = GridSearchCV(svm.SVC(), parameters, cv=4, verbose=2)

model.fit(X_train, y_train)
# params = model.best_params_
# score = model.score(train_X, train_Y)
pred_y = model.predict(X_test)
# print(params, score)
pd.set_option('display.max_columns', None)
result = pd.DataFrame.from_dict(model.cv_results_)
print(result)

score = model.score(X_train, y_train)
print("score: Score = %f" % score)
test_score = model.score(X_test, y_test)
print("test_score: Score = %f" % test_score)
precision = precision_score(pred_y, y_test, average='weighted')
print("precision: Score = %f" % precision)
recall = recall_score(pred_y, y_test, average='weighted')
print("recall: Score = %f" % recall)
f1_score = f1_score(pred_y, y_test, average='weighted')
print("f1_score: Score = %f" % f1_score)

# output
# svc = svm.SVC(kernel='linear', C=1, verbose=2)
# svc.fit(train_X, train_Y)
# predict_Y = svc.predict(test_X)
# score = svc.score(train_X, train_Y)
# print("SVM score: Score = %f" % score)
# out = pd.DataFrame(test.loc[:, "Id"], columns=["Id"])
# out.insert(1, "Response", predict_Y)
# out.to_csv("Submission/SVM.csv", index=False)

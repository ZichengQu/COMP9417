from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import accuracy_score, classification_report

import preprocessing

train_X, train_Y, test_X, test_Y, train, test = preprocessing.getDataSet(dataSet = 5) # shape: (59381, 122), (59381,), (19765, 122), None, (59381, 128), (19765, 127)

# LINEAR Regression
clf = LinearRegression()
clf.fit(train_X,train_Y)
print('Params set is: ', clf.get_params())
print('Score is: ', clf.score(train_X,train_Y)) # The lower the score, the worse the model is.
# print('Train Response Predict: ',clf.predict(train_X))
# print('Coefficient is: ',clf.coef_)
predict_train_Y = clf.predict(train_X)
predict_test_Y = clf.predict(test_X)
predict_train_Y = predict_train_Y.astype(int)
predict_test_Y = predict_test_Y.astype(int)
print('Train predict accuracy is', accuracy_score(train_Y, predict_train_Y))
print(classification_report(train_Y, predict_train_Y, zero_division=0))

# Ridge Regression
lamda = [0.01,0.1,0.5,1,1.5,2,5,10,20,30,50,100,200,300]
acc_Ridge=[]
for i in range(len(lamda)):
    Reg = Ridge(lamda[i])
    ret = Reg.fit(train_X, train_Y)
    Reg.score(train_X, train_Y)
    predict_Y = Reg.predict(train_X)
    predict_Y = predict_Y.astype(int)
    acc_Ridge.append(accuracy_score(train_Y, predict_Y))

# Lasso Regression
acc_Lasso=[]
for i in range(len(lamda)):
    Las = Lasso(lamda[i])
    ret = Las.fit(train_X, train_Y)
    Las.score(train_X, train_Y)
    predict_Y = Las.predict(train_X)
    predict_Y = predict_Y.astype(int)
    acc_Lasso.append(accuracy_score(train_Y, predict_Y))

print('Ridge Regression',acc_Ridge)
print('Lasso Regression',acc_Lasso)
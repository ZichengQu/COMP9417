import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

import preprocessing

# Get the data set from the preprocessing.py.
train_X, train_Y, test_X, test_Y, train, test = preprocessing.getDataSet(dataSet = 4) # shape: (59381, 89), (59381,), (19765, 89), None, (59381, 128), (19765, 127)

# Split the data set via train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_X, train.loc[:, "Response"], random_state=0, test_size=0.3)


param_grid = [
    {
        'weights': ['uniform'],
        'n_neighbors': [8]
    },
    # {
    #     'weights': ['distance'],
    #     'n_neighbors': [100],
    #     'p': [2]
    # }
]

knn = KNeighborsClassifier()

model = GridSearchCV(knn, param_grid, cv=4, verbose=2)

model.fit(X_train, y_train)
params = model.best_params_
print("best params:", params)
pred_y = model.predict(X_test)

pd.set_option('display.max_columns', None)
result = pd.DataFrame.from_dict(model.cv_results_)
print(result)

score = model.score(X_train, y_train)
print("KNN score: Score = %f" % score)
test_score = model.score(X_test, y_test)
print("KNN test_score: Score = %f" % test_score)
precision = precision_score(pred_y, y_test, average='weighted')
print("KNN precision: Score = %f" % precision)
recall = recall_score(pred_y, y_test, average='weighted')
print("KNN recall: Score = %f" % recall)
f1_score = f1_score(pred_y, y_test, average='weighted')
print("KNN f1_score: Score = %f" % f1_score)


# output
# starttime = datetime.now()
# knn = KNeighborsClassifier(weights='uniform', n_neighbors=27)
# knn.fit(train_X, train_Y)
# predict_y = knn.predict(test_X)
# endtime = datetime.now()
# out = pd.DataFrame(test.loc[:, "Id"], columns=["Id"])
# out.insert(1, "Response", predict_y)
# out.to_csv("Submission/knn.csv", index=False)
# print((endtime - starttime).seconds)
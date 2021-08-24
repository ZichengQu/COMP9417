import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import figure_plot
import test_grid
import preprocessing

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
import pickle

"""
The main idea:
1. Get the data set from the Feature Engineering (preprocessing.py).
2. Set the params for MLPClassifier and BaggingClassifier.
3. Use GridSearchCV to help find global best params.
4. There is no test_Y provided in Kaggle, use KFold (Cross Validation) to evaluate and test the model.
5. Train and Test the model.
6. Save to csv.
7. Save or load the model if needed.
"""

isPlot = True # Whether to plt (saved in "./figures")

# 1. Get the data set from the preprocessing.py.
train_X, train_Y, test_X, test_Y, train, test = preprocessing.getDataSet(dataSet = 1) # shape: (59381, 89), (59381,), (19765, 89), None, (59381, 128), (19765, 127)

# 2. Set the params for MLPClassifier and BaggingClassifier.
params_mlp = {"max_iter": 500, "activation": 'tanh', "early_stopping": False, "hidden_layer_sizes": (16, 64, 128, 64, 32, 8), "shuffle": True, "solver": 'adam', "verbose": False}
params_bagging = {"base_estimator": MLPClassifier(**params_mlp), "n_estimators": 10, "max_samples": 0.99, "max_features": 0.99, "verbose": 10}

# 3. Use GridSearchCV to help find global best params. GridSearchCV is defined in another file named test_grid.py
# classifier = test_grid.grid_search(train_X, train_Y, params)

def kFold(params_bagging, n_splits):
    '''
    KFold (cross validation)
    In this model, 10 Fold with BaggingClassifier, each BaggingClassifier contains 10 MLPClassifier.
    '''
    kf = KFold(n_splits = n_splits)
    temp_train = np.hstack((train_X, np.array(train_Y).reshape(-1, 1))) # shape: (59381, 90)
    train_score_list = list()
    test_score_list = list()
    progress = 0
    for KFold_train, KFold_test in kf.split(temp_train):
        print("Current progress: ", progress * 100 / n_splits, "%")
        progress += 1
        KFold_train_X = temp_train[KFold_train][:, :-1]
        KFold_train_Y = temp_train[KFold_train][:, -1]
        KFold_test_X = temp_train[KFold_test][:, :-1]
        KFold_test_Y = temp_train[KFold_test][:, -1]
        mlp = BaggingClassifier(**params_bagging)
        mlp.fit(KFold_train_X, KFold_train_Y)
        scoreTrain = mlp.score(KFold_train_X, KFold_train_Y)
        scoreTest = mlp.score(KFold_test_X, KFold_test_Y)
        train_score_list.append(scoreTrain)
        test_score_list.append(scoreTest)
    # print(train_score_list) # [0.5643127128475731, 0.5684561121194543, 0.5788035851280804, 0.5800385457403214, 0.5759594334150403, 0.5766330482944445, 0.5686058043148775, 0.5597178302116274, 0.560485002713171, 0.5679696124843291]
    # print(test_score_list) # [0.5137228489644722, 0.5062310542270124, 0.5126305153250252, 0.5183563489390367, 0.5040417649040081, 0.5053890198720108, 0.5070730885820142, 0.5050522061300101, 0.5099360053890198, 0.5003368137420007]
    if isPlot:
        figure_plot.plot_train_test_score(n_splits, train_score_list, test_score_list)

# 4. There is no test_Y provided in Kaggle, hence use KFold (Cross Validation) to test the model.
# kFold(params_bagging = params_bagging, n_splits = 10)

# 5. Train and Test the model
classifier = BaggingClassifier(**params_bagging)
classifier.fit(train_X, train_Y)
predicted_test_Y = classifier.predict(test_X)

# 6. save to csv
submission = pd.DataFrame(test.loc[:, "Id"], columns=["Id"])
submission.insert(1, "Response", predicted_test_Y)
submission.to_csv("submission/submission.csv", index=False)

#save the model
def save_model(classifier, file_name):
    file = open(file_name, 'wb')
    pickle.dump(classifier, file)
    file.close()

# load the model
def load_model(file_name):
    file = open(file_name, 'rb')
    load_classifier = pickle.load(file)
    file.close()
    return load_classifier

# 7. Save of load the model if needed
save_model(classifier, "submission/classifier.pickle")
# classifier = load_model("submission/classifier.pickle")
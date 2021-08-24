# COMP9417 Project: Prudential Life Insurance Assessment

## 1. <a href="https://www.kaggle.com/c/prudential-life-insurance-assessment/" target="_blank"><b>`Kaggle Link:` Prudential Life Insurance Assessment</b></a><br>

## 2. **[`Feature Engineering` in preprocessing.py](preprocessing.py)**
The main process:
1. Read the data set from train and test.
2. Classify the features according to the data types prompted in Kaggle.
3. Split or reconstruct some features, and update the data set.
4. If the missing ratio is large for some feature, then remove the features.
5. For the features with small missing ratio, mode value of each feature is used to fill in.
6. Removes features with high-frequency single value occurrence.
7. Identify features with high positive or negative correlations via pairplot, and remove some of them.
8. Verify the correlation again with the heatmap, decrease the correlation.

## 3. **[`Final Model BaggingClassifier with MLPClassifier` in model_MLP.py](model_MLP.py)**
This model is the final one we choosed. </br>
The main idea:
1. Get the data set from the **[Feature Engineering (preprocessing.py)](preprocessing.py)**.
2. Set the params for MLPClassifier and BaggingClassifier.
3. Use GridSearchCV to help find global best params.
4. There is no test_Y provided in Kaggle, use KFold (Cross Validation) to evaluate and test the model.
5. Train and Test the model.
6. Save to csv.
7. Save or load the model if needed.

## 4. Other Models
### All other models we used in this project are listed below, but were not chosen as the final model.

- ### **[`LinearRegression` in model_LinearRegression.py](model_LinearRegression.py)**

- ### **[`DecisionTree` in model_DecisionTree.py](model_DecisionTree.py)**

- ### **[`LogisticRegression` in model_LogisticRegression.py](model_LogisticRegression.py)**

- ### **[`BernoulliNB and MultinomialNB` in model_NB.py](model_NB.py)**

- ### **[`KNN` in model_KNN.py](model_KNN.py)**

- ### **[`SVC` in model_linearSVC.py.py](model_linearSVC.py.py)**

- ### **[`SVM` in model_SVM.py.py](model_SVM.py.py)**

## 5. Util Files
- ### **[`Figure` in figure_plot.py](figure_plot.py)** </br> 
#### <pre>The functions in [figure_plot.py](figure_plot.py) will be called in Preprocessing and MLP to plot figures.</pre>
- ### **[`GridSearchCV` in test_grid.py](test_grid.py)** </br>
#### <pre>The functions in [test_grid.py](test_grid.py) will be called in MLP to seek the best params.</pre>


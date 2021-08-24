import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import figure_plot

"""
The main idea:
1. Read the data set from train and test.
2. Classify the features according to the data types prompted in Kaggle.
3. Split or reconstruct some features, and update the data set.
4. If the missing ratio is large for some feature, then remove the features.
5. For the features with small missing ratio, mode value of each feature is used to fill in.
6. Removes features with high-frequency single value occurrence.
7. Identify features with high positive or negative correlations via pairplot, and remove some of them.
8. Verify the correlation again with the heatmap, decrease the correlation.
"""

isPlot = False # Whether to plt (saved in "./figures")

# 1. Read the data set from train and test.
train = pd.read_csv("DataSet/train.csv")
test = pd.read_csv("DataSet/test.csv")

if isPlot:
    figure_plot.labelDistribution(train) # The label 'Response' distribution

# Initialize the Train Set
train_X = train.iloc[:, 1:-1] # shape (59381, 126)
train_Y = train.loc[:, "Response"]

# Initialize the Test Set
test_X = test.iloc[:, 1:] # shape (19765, 126)
test_Y = None  # Invisible in Kaggle. Use KFold to split the train set to predict and adjust parameters.

# 2. Classify the features according to the data types prompted in Kaggle.
categorical_col = ["Product_Info_" + str(i) for i in [1, 2, 3, 5, 6, 7]] + \
    ["Employment_Info_" + str(i) for i in [2, 3, 5]] + \
    ["InsuredInfo_" + str(i) for i in range(1, 8)] + \
    ["Insurance_History_" + str(i) for i in [1, 2, 3, 4, 7, 8, 9]] + \
    ["Family_Hist_1"] + \
    ["Medical_History_" + str(i)for i in range(2, 42) if i not in [10, 15, 24, 32]]

continuous_col = ["Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI"] + \
    ["Employment_Info_" + str(i) for i in [1, 4, 6]] + \
    ["Insurance_History_5"] + \
    ["Family_Hist_" + str(i) for i in range(2, 6)]

discrete_col = ["Medical_History_" + str(i) for i in [1, 10, 15, 24, 32]]

dummy_col = ["Medical_Keyword_" + str(i) for i in range(1, 49)]

# 3. Split or reconstruct some features, and update the data set.
train_X["Age_BMI"] = train_X["Ins_Age"] * train_X["BMI"] # Age_BMI = BMI * Age. The values provided in the Kaggle csv have already been transformed between 0 - 1.
train_X["Product_Info_2_char"] = train_X["Product_Info_2"].str[0] # Product_info_2 is combined of charactor and number. Split the char and number into two new features.
train_X["Product_Info_2_num"] = train_X["Product_Info_2"].str[1]
train_X["Product_Info_2_char"] = pd.factorize(train_X["Product_Info_2_char"])[0] # factorize: mapping same char => same number。
test_X["Age_BMI"] = test_X["Ins_Age"] * test_X["BMI"] # Same operation for the test set.
test_X["Product_Info_2_char"] = test_X["Product_Info_2"].str[0]
test_X["Product_Info_2_num"] = test_X["Product_Info_2"].str[1]
test_X["Product_Info_2_char"] = pd.factorize(test_X["Product_Info_2_char"])[0]
continuous_col.append("Age_BMI") # Age_BMI belongs to continuous features
categorical_col += ["Product_Info_2_char", "Product_Info_2_num"]
categorical_col.remove("Product_Info_2")

# Update the data set
train_X = train_X[categorical_col + continuous_col + discrete_col + dummy_col] # shape: (59381, 126) => (59381, 129) => (59381, 128)
test_X = test_X[categorical_col + continuous_col + discrete_col + dummy_col] # shape: (19765, 126) => (19765, 129) => (19765, 128)

def delete_cols(to_delete):
    '''
    Delete some features of the four types of features.
    to_delete: list(features to be removed).
    return: the four types of features.
    '''
    categorical_temp = list(set(categorical_col).difference(set(to_delete)))  # difference: list - to_delete
    continuous_temp = list(set(continuous_col).difference(set(to_delete)))
    discrete_temp = list(set(discrete_col).difference(set(to_delete)))
    dummy_temp = list(set(dummy_col).difference(set(to_delete)))
    return categorical_temp, continuous_temp, discrete_temp, dummy_temp

def findMissing(data, missing_rate):
    '''
    Find the features where the missing ratio is greater than the given rate.
    '''
    missing = pd.concat([data.count()], axis=1)
    missing = pd.DataFrame(index = list(missing.index), data = list(missing.values), columns = ['occurrence'])
    missing['missing rate']= (data.shape[0] - missing['occurrence']) / missing['occurrence'].max() # Missing ratio = (total row - non null row for this feautre) / total row
    # print(missing[missing['missing rate'] != 0])
    return list(missing[missing['missing rate'] > missing_rate].index)

# Delete features that are more than 50% missing
to_delete = findMissing(data = train_X, missing_rate = 0.5) 
categorical_col, continuous_col, discrete_col, dummy_col = delete_cols(to_delete)

# Update the data set
train_X = train_X[categorical_col + continuous_col + discrete_col + dummy_col] # shape: (59381, 128) => (59381, 122)
test_X = test_X[categorical_col + continuous_col + discrete_col + dummy_col] # shape: (19765, 128) => (19765, 122)

# Fill in the missing values with the mode of each feature
train_X.fillna(value = dict([(col, train_X[col].mode()[0]) for col in train_X.columns]) , inplace = True)
test_X.fillna(value = dict([(col, test_X[col].mode()[0]) for col in test_X.columns]) , inplace = True)
# The student who is responsible for LinearRegression and DecisionTree found it is better to use the current data set, so leave train_X and test_X here.
train_X_LinearRegression = train_X.copy(deep = True)
test_X_LinearRegression = test_X.copy(deep = True)

# Plot the frequency of the value occurrence for each feature
if isPlot:
    figure_plot.showOccurrences(train_X, categorical_col, "categorical_col")
    figure_plot.showOccurrences(train_X, continuous_col, "continuous_col")
    figure_plot.showOccurrences(train_X, discrete_col, "discrete_col")
    figure_plot.showOccurrences(train_X, dummy_col, "dummy_col")

def deleteHighFrequency(data, frequency):
    '''
    Get the features whose mode ratio exceeds "frequency"
    '''
    to_delete = list()
    for col in data.columns:
        ser = data[col]  # get the series
        mode = ser.mode()  # mode
        if (ser.value_counts()[mode] / len(ser) * 100).item() > frequency:
            to_delete.append(col)
    return to_delete

to_delete = deleteHighFrequency(train_X, 99)  # Get and delete the features whose mode ratio exceeds 99%
categorical_col, continuous_col, discrete_col, dummy_col = delete_cols(to_delete)
# Update the data set
train_X = train_X[categorical_col + continuous_col + discrete_col + dummy_col] # shape: (59381, 122) => (59381, 101)
test_X = test_X[categorical_col + continuous_col + discrete_col + dummy_col] # shape: (19765, 122) => (19765, 101)

# Plot the correlations via pairplot
if isPlot:
    # figure_plot.getPairplot(train_X, categorical_col, "categorical") # no help
    figure_plot.getPairplot(train_X, continuous_col, "continuous")
    figure_plot.getPairplot(train_X, discrete_col, "discrete")
    # figure_plot.getPairplot(train_X, dummy_col, "dummy") # no help

to_delete = ["Family_Hist_4", "Ins_Age", "BMI"]
categorical_col, continuous_col, discrete_col, dummy_col = delete_cols(to_delete)
# Update the data set
train_X = train_X[categorical_col + continuous_col + discrete_col + dummy_col] # shape: (59381, 101) => (59381, 98)
test_X = test_X[categorical_col + continuous_col + discrete_col + dummy_col] # shape: (19765, 101) => (19765, 98)

# Plot the correlations via heatmap, since some features are not suitable in pairplot
if isPlot:
    figure_plot.getHeatMap(train_X, categorical_col, "categorical_col")
    figure_plot.getHeatMap(train_X, continuous_col, "continuous_col")
    figure_plot.getHeatMap(train_X, discrete_col, "discrete_col")
    figure_plot.getHeatMap(train_X, dummy_col, "dummy_col")

to_delete = ["Medical_History_1", "Employment_Info_3", "Medical_History_25", "Medical_History_36", "Product_Info_3", "Insurance_History_3", "Insurance_History_4", "Insurance_History_7", "Insurance_History_9"]
categorical_col, continuous_col, discrete_col, dummy_col = delete_cols(to_delete)
# Update the data set
train_X = train_X[categorical_col + continuous_col + discrete_col + dummy_col] # shape: (59381, 98) => (59381, 89)
test_X = test_X[categorical_col + continuous_col + discrete_col + dummy_col] # shape: (19765, 98) => (19765, 89)

def getDataSet(dataSet = 0, train_X = train_X, test_X = test_X, train = train, test = test):
    if dataSet == 0: # after preprocessing, use all features left
        train_X = train_X # shape: (59381, 89)
        test_X = test_X # shape: (19765, 89)
    elif dataSet == 1: # after preprocessing, use all features left, and do Normalization
        train_X = normalize(np.array(train_X), norm='l2')
        test_X = normalize(test_X, norm='l2')
    elif dataSet == 2: # Discard Dummy Variables
        train_X = train_X[categorical_col + continuous_col + discrete_col] # shape: (59381, 89) => (59381, 54)
        test_X = test_X[categorical_col + continuous_col + discrete_col] # shape: (19765, 89) => (19765, 54)
    elif dataSet == 3:
        minMaxScaler = MinMaxScaler()
        train_X = minMaxScaler.fit_transform(train_X)
        test_X = minMaxScaler.fit_transform(test_X)
    elif dataSet == 4:
        standardScaler = StandardScaler()
        train_X = standardScaler.fit_transform(train_X)
        test_X = standardScaler.fit_transform(test_X)
    elif dataSet == 5: # used in LinearRegression
        train_X = train_X_LinearRegression
        test_X = test_X_LinearRegression
    else: # more combinations
        pass

    # Convert to NumPy
    train_X = np.array(train_X) # shape: (59381, ?) # ? depends on the chosen dataSet above
    # train_Y = np.array(train_Y).reshape(-1, 1) # shape: (59381, 1)
    test_X = np.array(test_X) # shape: (19765, ?)
    test_Y = None # Not provided in Kaggle

    return train_X, train_Y, test_X, test_Y, train, test
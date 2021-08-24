import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

# Question 2 (a)
df = pd.read_csv("data.csv")
df_X = df.iloc[:,:-1]
sns.pairplot(df_X)
plt.savefig("Question 2 (a).png")


# Question 2 (b)
df_X = df_X - df_X.mean()

df_X = df_X * (np.sqrt(df_X.shape[0]) / np.sqrt(np.power(df_X, 2).sum()))

print(np.power(df_X, 2).sum())


# Question 2 (c)
df_Y = df.iloc[:,-1]
coef = list()

alpha_range = [0.01, 0.1, 0.5, 1, 1.5, 2, 5, 10, 20, 30, 50, 100, 200, 300]
colors = ['red', 'brown', 'green', 'blue', 'orange', 'pink', 'purple', 'grey']
for alpha in alpha_range:
    ridge = Ridge(alpha=alpha)
    ridge.fit(df_X, df_Y)
    coef.append(ridge.coef_.tolist())

coef = np.array(coef)

plt.figure()
plt.xlabel("log(λ)")
plt.ylabel("Coefficient of Features")
for i in range(coef.shape[1]):
    plt.plot(np.log(alpha_range), coef[:, i], marker = "o", markersize=3, color= colors[i])
    if i == 5:
        plt.text(np.log(alpha_range[0]), coef[0, i] - 1.5, 'X'+str(i+1)) # Overlap with i == 6
    else:
        plt.text(np.log(alpha_range[0]), coef[0, i], 'X'+str(i+1))
plt.savefig("Question 2 (c).png")


# Question 2 (d)
alpha_range = np.arange(0, 50.1, 0.1)
MSEs = list()
for alpha in alpha_range:
    # print("process: " + str(alpha * 100 // 50) + "%")
    MSE_ridge = 0
    for row in range(df.shape[0]): # Leave-One-Out Cross Validation for Ridge
        # Test
        test_X = df_X.loc[row].values.reshape(1, -1) # Convert Series to NumPy and then reshape
        test_Y = df.iloc[row, -1]
        # Train
        train_X = df_X.iloc[:row]
        train_X = train_X.append(df_X.iloc[row+1:])
        train_Y = df.iloc[:row, -1]
        train_Y = train_Y.append(df.iloc[row+1:, -1])

        # Get MSE for Ridge with different λ
        ridge = Ridge(alpha=alpha)
        ridge.fit(train_X, train_Y)
        predicted_Y = ridge.predict(test_X)
        predicted_Y = predicted_Y[0] # From np to integer
        MSE_ridge += pow((test_Y - predicted_Y), 2) # This should be squared error, not mean  squared error
    MSE_ridge /= df.shape[0] # Real MSE
    MSEs.append(MSE_ridge)

alpha = alpha_range[MSEs.index(min(MSEs))] # The λ when taking the min(MSEs)
print("MSE for Ridge: {}, The λ is {}".format(min(MSEs), alpha)) # MSE for Ridge: 1442.6982227952913, The λ is 22.3

plt.figure()
plt.xlabel("λ")
plt.ylabel("MSE")
plt.plot(alpha_range, MSEs)
MSE_min_row = alpha_range[MSEs.index(min(MSEs))] # x for the min(MSEs)
MSE_min_col = int(min(MSEs) * 100) / 100 # y for the min(MSEs)
plt.plot(MSE_min_row, MSE_min_col, marker = "o", markersize=3) # label the point of the min(MSEs)
plt.text(MSE_min_row, MSE_min_col, "min(MSE):("+str(MSE_min_row)+", "+str(MSE_min_col)+")")
plt.savefig("Question 2 (d).png")

MSE_OLS = 0
for row in range(df.shape[0]): # Leave-One-Out Cross Validation for OLS
    # Test
    test_X = df_X.loc[row].values.reshape(1, -1) # Convert Series to NumPy and then reshape
    test_Y = df.iloc[row, -1]
    # Train
    train_X = df_X.iloc[:row]
    train_X = train_X.append(df_X.iloc[row+1:])
    train_Y = df.iloc[:row, -1]
    train_Y = train_Y.append(df.iloc[row+1:, -1])

    # Get squared error for OLS
    OLS = LinearRegression()
    OLS.fit(train_X, train_Y)
    predicted_Y = OLS.predict(test_X)
    predicted_Y = predicted_Y[0] # From np to integer
    MSE_OLS += pow((test_Y - predicted_Y), 2) # This should be squared error, not mean squared error

MSE_OLS /= df.shape[0] # Real MSE
print("MSE for LinearRegression: ", MSE_OLS) # MSE for LinearRegression:  1975.4147393421736

# Question 2 (e)
df_Y = df.iloc[:,-1]
coef = list()

alpha_range = [0.01, 0.1, 0.5, 1, 1.5, 2, 5, 10, 20, 30, 50, 100, 200, 300]
colors = ['red', 'brown', 'green', 'blue', 'orange', 'pink', 'purple', 'grey']
for alpha in alpha_range:
    lasso = Lasso(alpha=alpha)
    lasso.fit(df_X, df_Y)
    coef.append(lasso.coef_.tolist())

coef = np.array(coef)

plt.figure()
plt.xlabel("log(λ)")
plt.ylabel("Coefficient of Features")
for i in range(coef.shape[1]):
    plt.plot(np.log(alpha_range), coef[:, i], marker = "o", markersize=3, color= colors[i])
    if i == 5:
        plt.text(np.log(alpha_range[0]), coef[0, i] - 1.5, 'X'+str(i+1)) # Overlap with i == 6
    else:
        plt.text(np.log(alpha_range[0]), coef[0, i], 'X'+str(i+1))
plt.savefig("Question 2 (e).png")

# Question 2 (f)
alpha_range = np.arange(0, 20.1, 0.1)
MSEs = list()
for alpha in alpha_range:
    # print("process: " + str(alpha * 100 // 20) + "%")
    MSE_lasso = 0
    for row in range(df.shape[0]): # Leave-One-Out Cross Validation
        # Test
        test_X = df_X.loc[row].values.reshape(1, -1) # Convert Series to NumPy and then reshape
        test_Y = df.iloc[row, -1]
        # Train
        train_X = df_X.iloc[:row]
        train_X = train_X.append(df_X.iloc[row+1:])
        train_Y = df.iloc[:row, -1]
        train_Y = train_Y.append(df.iloc[row+1:, -1])

        # Get MSE for Lasso with different λ
        lasso = Lasso(alpha=alpha)
        lasso.fit(train_X, train_Y)
        predicted_Y = lasso.predict(test_X)
        predicted_Y = predicted_Y[0] # From np to integer
        MSE_lasso += pow((test_Y - predicted_Y), 2) # This should be squared error, not mean  squared error
    MSE_lasso /= df.shape[0] # Real MSE
    MSEs.append(MSE_lasso)

plt.figure()
plt.xlabel("λ")
plt.ylabel("MSE")
plt.plot(alpha_range, MSEs)
MSE_min_row = alpha_range[MSEs.index(min(MSEs))] # x for the min(MSEs)
MSE_min_col = int(min(MSEs) * 100) / 100 # y for the min(MSEs)
plt.plot(MSE_min_row, MSE_min_col, marker = "o", markersize=3) # label the point of the min(MSEs)
plt.text(MSE_min_row, MSE_min_col, "min(MSE):("+str(MSE_min_row)+", "+str(MSE_min_col)+")")
plt.savefig("Question 2 (f).png")

alpha = alpha_range[MSEs.index(min(MSEs))] # The λ when taking the min(MSEs)
print("MSE for LASSO: {}, The λ is {}".format(min(MSEs), alpha)) # MSE for LASSO: 1586.6715081806426, The λ is 5.5


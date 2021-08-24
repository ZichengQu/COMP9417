import numpy as np
from numpy.core.fromnumeric import argmin
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from sklearn.preprocessing import MinMaxScaler
import jax.numpy as jnp
from jax import grad

from scipy.optimize import minimize

#####################################################################
# Question 1 (b)
#####################################################################
data = pd.read_csv("Q1.csv")

train = data[:500]
test = data[500:]

train_x = train.iloc[:,:45]
train_y = train.iloc[:,-1]

test_x = test.iloc[:,:45]
test_y = test.iloc[:,-1]

C_grid = np.linspace(0.0001, 0.6, 100)
record = list() # Record log-loss of the kFold for the 100 C values
for c in C_grid:
    kFold = 10
    sub_record = list()
    for i in range(kFold):
        # define model
        classifier = LogisticRegression(C = c, penalty = "l1", solver= "liblinear", random_state = 0)

        # process dataset
        test_start = 50 * i
        test_end = test_start + 50

        train_grid = train[0: test_start].append(train[test_end:])
        test_grid = train[test_start: test_end]

        train_grid_x = train_grid.iloc[:,:45]
        train_grid_y = train_grid.iloc[:,-1]

        test_grid_x = test_grid.iloc[:,:45]
        test_grid_y = test_grid.iloc[:,-1]

        # fit and predict
        classifier.fit(train_grid_x, train_grid_y)
        predicted_grid_y = classifier.predict_proba(test_grid_x)

        # calculate log_loss in every loop of kFold
        logloss = log_loss(test_grid_y, predicted_grid_y)

        sub_record.append(logloss)

    record.append(sub_record)


record_mean = [sum(i) / len(i) for i in record] # calculate the mean value for each C
choosen_C = C_grid[argmin(record_mean)] # choose c with the min(mean)
print(choosen_C) # 0.18794747474747472

log_loss_result = pd.DataFrame(columns=['C', 'log_loss'])

for index, sub_record in enumerate(record):
    for loss in sub_record:
        log_loss_result.loc[log_loss_result.shape[0] - 1]=[C_grid[index], loss]

sns.boxplot(x="C", y="log_loss", data=log_loss_result)
plt.savefig("Question 1(b).png")
plt.clf()

# Re-fit the model with this chosen C, and get the accuracy for both train and test via this model.
classifier = LogisticRegression(C = choosen_C, penalty = "l1", solver= "liblinear", random_state = 0)
classifier.fit(train_x, train_y)
print(classifier.score(train_x, train_y)) # 0.752
print(classifier.score(test_x, test_y)) # 0.74

#####################################################################
# Question 1 (c)
#####################################################################
C_grid = np.linspace(0.0001, 0.6, 100)
grid_lr = GridSearchCV(estimator = LogisticRegression(penalty='l1',solver='liblinear'),
                        cv = KFold(n_splits=10),
                        scoring = 'neg_log_loss',
                        param_grid = {'C': C_grid})

grid_lr.fit(train_x, train_y)

print(grid_lr.best_params_) # original: {'C': 0.012219191919191918}, after modified: {'C': 0.18794747474747472}

#####################################################################
# Question 1 (d)
#####################################################################
np.random.seed(12)
c = 1
coefs = pd.DataFrame(columns=['X' + str(m) for m in range(train.shape[1] - 1)])
for i in range(10000):
    if i % 50 == 0:
        print("Process: " + str(i / 100) + "%")
    row = [np.random.choice(500) for j in range(500)] # 500 random index from [0, 499]
    train_bootstrap = train.loc[row] # train set for bootstrap
    train_bootstrap_x = train_bootstrap.iloc[:,:45]
    train_bootstrap_y = train_bootstrap.iloc[:,-1]

    classifier = LogisticRegression(C = c, penalty = "l1", solver= "liblinear", random_state = 0)
    classifier.fit(train_bootstrap_x, train_bootstrap_y)
    coefs.loc[coefs.shape[0] - 1] = np.squeeze(classifier.coef_, axis=0) # Append coef to coefs

quantile_5 = list()
quantile_95 = list()
avg = list()
for index, col in coefs.iteritems():
    quantile_5.append(np.percentile(sorted(col), 5))
    quantile_95.append(np.percentile(sorted(col), 95))
    avg.append(sum(col) / len(col))
    
x_lable = [i for i in range(45)]
colors_list = ['red' if quantile_5[i] * quantile_95[i] <= 0 else 'blue' for i in range(45)]
plt.vlines(x = x_lable, ymin = quantile_5, ymax = quantile_95, lw = 3, colors = colors_list, linestyles = '-')
plt.scatter(x_lable, avg, c= "black")
plt.xlabel("parameters β")
plt.ylabel("5th to 95th quantile of bootstrap estimates")
plt.savefig("Question 1(d).png")
plt.clf()

#####################################################################
# Question 2 (a)
#####################################################################
A = np.array([[1, 0, 1, -1],
            [-1, 1, 0, 2],
            [0, -1, -2, 1]])
b = np.array([[1],
            [2],
            [3]])
x = np.array([[1],
            [1],
            [1],
            [1]])
lr = 0.1

f = 0.5 * np.dot((np.dot(A, x) - b).T, (np.dot(A, x) - b)) # shape = (1, 1)
derivation_f = np.dot(A.T, (np.dot(A, x) - b)) # f's derivation

x_list = list() # to record the x for each iteration
# x_list.append(x) # I suppose the initial one is not in the iteration, so I did not involve it in.
while True:
    if np.linalg.norm(derivation_f) < 0.001: # 2 norm
        break
    x = x - lr * derivation_f # Update x
    derivation_f = np.dot(A.T, (np.dot(A, x) - b)) # Update derivation
    x_list.append(x)

for i in range(5): # the first 5 values of x_k
    print(x_list[i].T) # To make it more intuitive, I print it out in the form of transpose

for i in range(-5, 0): # the last 5 values of x_k
    print(x_list[i].T) # To make it more intuitive, I print it out in the form of transpose

#####################################################################
# Question 2 (b)
#####################################################################
A = np.array([[1, 0, 1, -1],
            [-1, 1, 0, 2],
            [0, -1, -2, 1]])
b = np.array([[1],
            [2],
            [3]])
x = np.array([[1],
            [1],
            [1],
            [1]])
lr = 0.1

f = 0.5 * np.dot((np.dot(A, x) - b).T, (np.dot(A, x) - b)) # shape = (1, 1)
derivation_f = np.dot(A.T, (np.dot(A, x) - b)) # f's derivation

x_list = list() # to record the x for each iteration
lr_list = list() # to record the lr for each iteration
# x_list.append(x) # I suppose the initial one is not in the iteration, so I did not involve it in.
lr_list.append(lr)
while True:
    if np.linalg.norm(derivation_f) < 0.001: # 2 norm
        break
    x = x - lr * derivation_f # Update x
    derivation_f = np.dot(A.T, (np.dot(A, x) - b)) # Update derivation
    lr = (np.dot(np.dot(A, x).T, np.dot(A, derivation_f)) - np.dot(np.dot(A, derivation_f).T, b)) / np.dot(np.dot(A, derivation_f).T, np.dot(A, derivation_f))
    x_list.append(x)
    lr_list.append(lr[0][0])

for i in range(5): # the first 5 values of x_k
    print(x_list[i].T) # To make it more intuitive, I print it out in the form of transpose

for i in range(-5, 0): # the last 5 values of x_k
    print(x_list[i].T) # To make it more intuitive, I print it out in the form of transpose

plt.plot(range(1, len(lr_list) + 1), lr_list)
plt.xlabel("iterations")
plt.ylabel("learning rate")
plt.savefig("Question 2(b).png")
plt.clf()

#####################################################################
# Question 2 (d)
#####################################################################
data = pd.read_csv("Q2.csv")
data.dropna(axis=0, how='any', inplace=True)
X = data[["age", "nearestMRT", "nConvenience"]]
Y = data[["price"]]
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train = X[:int(X.shape[0] / 2)] # Train Set
Y_train = Y[:int(Y.shape[0] / 2)] # Train Label
X_test = X[int(X.shape[0] / 2):] # Test Set
Y_test = Y[int(Y.shape[0] / 2):] # Test Label

#####################################################################
# Question 2 (e)
#####################################################################
X_train_temp = np.insert(X_train, 0, values=1, axis=1) # Add a new first column with all the value 1.

def loss(w):
    loss = jnp.sum((jnp.sqrt(0.25 * jnp.square(jnp.array(Y_train) - jnp.dot(X_train_temp, w.T).reshape(-1, 1)) + 1) - 1)) / Y_train.shape[0]
    return loss

w = jnp.array([1.0,1.0,1.0,1.0])

iteration_counter = 0
lr = 1 # learning rate, step size 
training_loss_list = list()
while True:
    cur_loss = loss(w)
    training_loss_list.append(cur_loss)
    pre_loss = training_loss_list[len(training_loss_list) - 2] # when index < 0, take the first one, no exception occur.
    W_grad = grad(loss)(w)
    w = w - lr * W_grad # from the forum, Omar tells that to choose k if abs(loss_k - loss_(k-1)) < condition
    iteration_counter += 1
    if len(training_loss_list) > 1 and abs(cur_loss - pre_loss) < 0.0001: # the len(list) must be larger than 1
        break
print(iteration_counter) # len(training_loss_list) = iteration_counter = 1037
print(w) # [37.05697, -12.684172, -22.38835, 22.195482]
print("Train loss: ", training_loss_list[-1]) # 2.4737415

X_test_temp = np.insert(X_test, 0, values=1, axis=1) # Add a new first column with all the value 1.

def test_loss(w):
    loss = jnp.sum((jnp.sqrt(0.25 * jnp.square(jnp.array(Y_test) - jnp.dot(X_test_temp, w.T).reshape(-1, 1)) + 1) - 1)) / Y_test.shape[0]
    return loss
print("Test loss", test_loss(w)) # 2.6956608

plt.plot(range(len(training_loss_list)), training_loss_list)
plt.xlabel("iterations")
plt.ylabel("training loss")
plt.savefig("Question 2(e).png")
plt.clf()

#####################################################################
# Question 2 (f)
#####################################################################
X_train_temp = np.insert(X_train, 0, values=1, axis=1) # Add a new first column with all the value 1.

def loss(w):
    loss = jnp.sum((jnp.sqrt(0.25 * jnp.square(jnp.array(Y_train) - jnp.dot(X_train_temp, w.T).reshape(-1, 1)) + 1) - 1)) / Y_train.shape[0]
    return loss

def lr_func(args):
    old_w, W_grad = args
    # argmin L(w_k - lr * W_grad), L => loss function
    v = lambda lr: (sum((np.sqrt(0.25 * np.square(np.array(Y_train) - np.dot(X_train_temp, (old_w - lr * W_grad).T).reshape(-1, 1)) + 1) - 1)))[0] / Y_train.shape[0]
    return v

w = jnp.array([1.0,1.0,1.0,1.0])

iteration_counter = 0
lr = 1 # learning rate, step size 
training_loss_list = list()
while True:
    w_k = np.array(w) # lr need w_k, hence w_k should be stored before updated to w_(k+1)
    cur_loss = loss(w) # the current loss
    training_loss_list.append(cur_loss)
    W_grad = grad(loss)(w)
    w = w - lr * W_grad # Update the w_k to w_(k+1)

    args = (w_k, np.array(W_grad))
    lr_start = np.asarray((lr))
    optimizer = minimize(lr_func(args), lr_start, method='BFGS')
    lr = optimizer.x

    iteration_counter += 1
    if cur_loss < 2.5: # the len(list) must be larger than 1
        break

print(iteration_counter) # len(training_loss_list) = iteration_counter = 18
print(w) # [37.36413, -13.3802595, -20.870495, 21.694393]
print("Train loss: ", training_loss_list[-1]) # 2.4907434

X_test_temp = np.insert(X_test, 0, values=1, axis=1) # Add a new first column with all the value 1.

def test_loss(w):
    loss = jnp.sum((jnp.sqrt(0.25 * jnp.square(jnp.array(Y_test) - jnp.dot(X_test_temp, w.T).reshape(-1, 1)) + 1) - 1)) / Y_test.shape[0]
    return loss
print("Test loss", test_loss(w)) # 2.7091722

plt.plot(range(len(training_loss_list)), training_loss_list)
plt.xlabel("iterations")
plt.ylabel("training loss")
plt.savefig("Question 2(f).png")
plt.clf()
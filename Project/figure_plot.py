import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def labelDistribution(train):
    '''
    The label 'Response' distribution
    '''
    x = list(train.loc[:, "Response"].value_counts().index)
    height = train.loc[:, "Response"].value_counts().values
    plt.bar(x = x, height = height)
    plt.xlabel("Response")
    plt.ylabel("Occurrences")
    plt.savefig("figures/Response_distribution.png")


def showOccurrences(train_X, columns, name):
    '''
    Observe the frequency of the value occurrence for each feature
    '''
    temp_data = train_X[columns]
    ncols = len(temp_data.columns)
    fig = plt.figure(figsize = (4 * 5, (int(ncols / 4) + 1) * 6))
    for i, col in enumerate(temp_data.columns):
        nonnull_data = temp_data[col].dropna()
        plt.subplot(int(ncols / 4) + 1, 4, i + 1)
        plt.hist(nonnull_data, edgecolor = "black")
        plt.title(col, x = 0.7, y = 0.7)
    fig.tight_layout()
    plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = None, hspace = 0.99)
    plt.savefig("figures/showOccurrences_{}.png".format(name))

def getPairplot(train_X, columns, name):
    '''
    Show the correlations via pairplot
    '''
    sns.pairplot(train_X[columns], kind = "reg")
    plt.savefig("figures/pairplot_" + name + ".png")

def getHeatMap(train_X, columns, name):
    '''
    Show the correlations via heatmap, since some features are not suitable for pairplot
    '''
    temp_data = train_X[columns]
    plt.figure(figsize = [20,15])
    mask = np.zeros_like(temp_data.corr(), dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(temp_data.corr(), mask = mask, cmap = sns.diverging_palette(250, 10, as_cmap = True), linewidths = 0.5) # .5
    plt.title("HeatMap for " + str(name), fontsize = 15)
    plt.savefig("figures/HeatMap_{}.png".format(name))
    plt.clf()

def plot_train_test_score(n_splits, train_score_list, test_score_list):
    '''
    Show the Train and Test score with KFold via BaggingClassifier
    '''
    plt.plot(range(0, n_splits), train_score_list, label = "Train Set")
    plt.plot(range(0, n_splits), test_score_list, label = "Test Set")
    plt.legend()
    plt.title("Score for Train and Test with BaggingClassifier")
    plt.xlabel("KFold", size = "12")
    plt.ylabel("score", size = "12")
    plt.savefig("figures/KFold.png")
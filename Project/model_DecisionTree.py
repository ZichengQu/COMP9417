from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import preprocessing

train_X, train_Y, test_X, test_Y, train, test = preprocessing.getDataSet(dataSet = 5) # shape: (59381, 122), (59381,), (19765, 122), None, (59381, 128), (19765, 127)

# Decision Tree
stopping_criterion=[500,400,300,200,100,50,30,20,10,8,6,4,2,1]
acc_DT=[]
for i in range(len(stopping_criterion)):
    clf=DecisionTreeClassifier(min_samples_leaf=stopping_criterion[i], criterion='entropy',random_state=0)
    clf.fit(train_X,train_Y)
    pred_Y=clf.predict(train_X)
    acc_DT.append(accuracy_score(train_Y,pred_Y))

plt.xlabel('Stopping Criterion')
plt.ylabel('Scores')
plt.title('Decision Tree changing stopping criterian')
x=[i for i in range(len(acc_DT))]
plt.bar(x,acc_DT)
for a, b in zip(x, acc_DT):
    plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=8)
plt.xticks(x, stopping_criterion, size='small')
plt.tight_layout()
plt.show()
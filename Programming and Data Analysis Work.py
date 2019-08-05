# import requests
# import pandas as pd
# from bs4 import BeautifulSoup
# armop = []
# rtop = []
# mega = []
# c = requests.get('https://www.rottentomatoes.com/critic/armond-white/movies?page='+str(3))
# r = c.content
# arm = BeautifulSoup(r, features='html.parser')
# list = arm.find_all('td', attrs = {'class': 'col-xs-6 col-sm-3'})
# for i in range(100):
#     if i%2 == 0:
#         temp = list[i]
#         t2 = temp.prettify().find('icon tiny rotten')
#         if t2 > 0:
#             armop.append(1) #1 represents a negative opinion
#         if t2 < 0:
#             armop.append(0)
#     if i % 2 == 1:
#         temp = list[i]
#         t3 = temp.find('span', attrs = {'class':'tMeterScore'}).text[0:2]
#         rtop.append(int(t3))
#
#
# titles = []
# raw = arm.find_all('td',  attrs = {'class':'col-xs-12 col-sm-6 critic-review-table__title-column'})
# for unit in raw:
#     z = unit.find('a').text
#     z = z.strip()
#     titles.append(z)
#
# x = pd.DataFrame([titles, armop, rtop])
# x = x.transpose()
# x.to_excel("output.xlsx")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

f = open('output5.txt').read()
x = f.split("\t")
titles = []
rt = []
guard = []
re = []
iw = []
tbo = []
dbo = []
ow = []
bud = []
armo = []
for i in range(40):
    temp = x[11+(10*i)]
    titles.append(temp)
    temp = x[12 + (10 * i)]
    rt.append(float(temp)/100)
    temp = x[13 + (10 * i)]
    guard.append(float(temp))
    temp = x[14 + (10 * i)]
    re.append(float(temp))
    temp = x[15 + (10 * i)]
    iw.append(float(temp))
    temp = x[16 + (10 * i)]
    tbo.append(float(temp)/1e6)
    temp = x[17 + (10 * i)]
    dbo.append(float(temp)/1e6)
    temp = x[18 + (10 * i)]
    ow.append(float(temp)/1e6)
    temp = x[19 + (10 * i)]
    bud.append(float(temp)/1e6)
    temp = x[20 + (10 * i)]
    armo.append(temp)

armon = []
for i in armo:
    armon.append(float(i[0]))

ratio = []
for i in range(40):
    t2 = dbo[i]/tbo[i]
    ratio.append(t2)

# armon = pd.DataFrame(armon)
# iw = pd.DataFrame(iw)
# re = pd.DataFrame(re)
# rt = pd.DataFrame(rt)
# titles = pd.DataFrame(titles)
# guard = pd.DataFrame(guard)

L2y = armon[0:30]
L2ytest = armon[30:40]
L2x = []
LX = []
L2test = []
theta = np.array([-50, 0.1, 5, 5]) # initial guess for theta's
#
#
for i in range(30):
    L2x.append([rt[i], ow[i], bud[i]])
    LX.append([rt[i], ow[i], bud[i]])

for i in range(30,40):
    L2test.append([rt[i], ow[i], bud[i]])

for k in LX:
    k.insert(0, 1)

def X_shuffle(X,y):
    ind = np.array([i for i in range(X.shape[1])])
    np.random.shuffle(ind)
    X2 = deepcopy(X)
    y2 = deepcopy(y)
    for i in range(X.shape[1]):
        X2[:,i] = X[:,ind[i]]
        y2[i] = y[ind[i]]
    return X2, y2


X = (np.array(LX)).transpose() # array containing training examples
X_test = (np.array(L2test)).transpose()
M = len(X.transpose())
x, y2 = X_shuffle(np.array(L2x), np.array(L2y))

x1 = x.transpose()[0,:]
x2 = x.transpose()[1,:]
x3 = x.transpose()[2,:]

z = np.dot(theta, X)


# #scatter plot the data
# plt.close('all')
# arg_0 = np.where(y == 0)
# arg_1 = np.where(y == 1)
#
# fig1 = plt.figure()
# ax = fig1.add_subplot(111, projection = '3d')
#
# passed = ax.scatter(xs = x1[arg_1], ys = x2[arg_1], zs =x3[arg_1], zdir = 'z', marker='o', color='b', label='passed')
# failed = ax.scatter(xs = x1[arg_0], ys = x2[arg_0], zs =x3[arg_0], zdir = 'z', marker='x', color='r', label='failed')
# ax.set_xlabel('RT'); ax.set_ylabel('Opening Weekend'); ax.set_zlabel('Budget')
# ax.set_title('Critics, Budgets, and AW Opinion')
# ax.legend([passed, failed], ['disliked', 'liked'], loc=2)
# plt.show(block=False)
# plt.close(fig1)

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#
X_train = x
X_test = L2test
X3 = L2x

scaler.fit(X_train)
X_train = scaler.transform(X_train)
clf = MLPClassifier(activation='relu', hidden_layer_sizes=(14,5,2), max_iter=1500, alpha=1e-3,
                    solver='sgd', random_state=1,
                    learning_rate_init=0.01)

clf.fit(X_train, y2)
print(f'{clf.predict(X_train)}: predicted')
print(f'{y2}: actual')
preds2 = clf.predict(X_train)
train_acc2 = 0
for i in range(len(preds2)):
    if preds2[i] == y2[i]:
        train_acc2 +=1
    else:
        continue
print(f'Neural Net Training accuracy = {(train_acc2/len(preds2))*100}%')


# testing
x_test, y_test = X_shuffle(np.array(X_test), np.array(L2ytest))

print(f'{clf.predict(x_test)}: predicted')
print(f'{y_test}: actual')
preds2_test = clf.predict(x_test)
test_acc2 = 0
for i in range(len(preds2_test)):
    if preds2_test[i] == y_test[i]:
        test_acc2 +=1
    else:
        continue
print(f'Neural Net Testing accuracy = {(test_acc2/len(preds2_test))*100}%')


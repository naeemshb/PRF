from sklearn.datasets import make_classification
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from RF import RandomForestClassifier as PRF
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import matplotlib.pyplot as plt
import random

warnings.filterwarnings("ignore")

def main_inject_noise(x, u):

    X = []
    for i in range(u.shape[0]):
        temp = []
        for j in range(u.shape[1]):
            temp.append(np.random.normal(x[i][j], u[i][j], 1)[0])

        X.append(temp)

    return np.array(X)


def change_label(y ,f):
    n = int(f * y.shape[0])
    inds = list(range(y.shape[0]))
    random.shuffle(inds)
    inds = inds[:n]

    for i in inds:
        if y[i] == 0:
            y[i] = 1
        else:
            y[i] = 0

    return y


def inject_noise1(X, Ns):
    nf = X.shape[1]
    ns = X.shape[0]

    rf = np.random.uniform(0,1,nf)
    rf = np.repeat(rf.reshape(1,-1), ns, axis = 0)

    rs = np.random.uniform(0,1,ns)
    rs = np.repeat(rs.reshape(1,-1), nf, axis = 0).T

    noise = rs*rf*Ns
    std = np.std(X, axis = 0)
    std = np.repeat(std.reshape(1,-1), ns, axis = 0)

    return noise * std

def inject_noise2(X, Ns):

    X1 = X[:int(X.shape[0]/2)]
    X2 = X[int(X.shape[0]/2):]

    nf1 = X1.shape[1]
    nf2 = X2.shape[1]

    ns1 = X1.shape[0]
    ns2 = X2.shape[0]

    rf1 = np.random.uniform(0,1,nf1)
    rf1 = np.repeat(rf1.reshape(1,-1), ns1, axis = 0)

    rf2 = np.random.uniform(0,1,nf2)
    rf2 = np.repeat(rf2.reshape(1,-1), ns2, axis = 0)

    rf = np.append(rf1, rf2, axis = 0)

    ns = X.shape[0]


    rs = np.random.uniform(0,1,ns)
    rs = np.repeat(rs.reshape(1,-1), X.shape[1], axis = 0).T

    noise = rs*rf*Ns
    std = np.std(X, axis = 0)
    std = np.repeat(std.reshape(1,-1), ns, axis = 0)

    return noise * std


def inject_noise3(X_train, X_test, Ns):

    X1 = X_train
    X2 = X_test

    X = np.append(X_train, X_test, axis = 0)

    nf1 = X1.shape[1]
    nf2 = X2.shape[1]

    ns1 = X1.shape[0]
    ns2 = X2.shape[0]

    rf1 = np.random.uniform(0, 1, nf1)
    rf1 = np.repeat(rf1.reshape(1, -1), ns1, axis=0)

    rf2 = np.random.uniform(0, 1, nf2)
    rf2 = np.repeat(rf2.reshape(1, -1), ns2, axis=0)

    rf = np.append(rf1, rf2, axis=0)

    ns = X.shape[0]

    rs = np.random.uniform(0, 1, ns)
    rs = np.repeat(rs.reshape(1, -1), X.shape[1], axis=0).T

    noise = rs * rf * Ns
    std = np.std(X, axis=0)
    std = np.repeat(std.reshape(1, -1), ns, axis=0)
    noise = noise * std
    return noise[:X_train.shape[0]], noise[X_train.shape[0]:]




X, y = make_classification(n_samples = 1000)


#Noise in the labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


fr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

prf_list = []
rf_list = []

for f in fr:

    prf = PRF()

    y_train_new = change_label(y_train.copy(), f)
    y_test_new = change_label(y_test.copy(), f)

    prf.fit(X_train, y_train_new)
    prf_predict = prf.predict(X_test)

    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(X_train, y_train_new)
    rf_predict = rf.predict(X_test)

    prf_score = accuracy_score(y_test_new, prf_predict)
    rf_score = accuracy_score(y_test_new, rf_predict)

    prf_list.append(prf_score)
    rf_list.append(rf_score)



plt.figure()
plt.plot(fr, prf_list, label = 'PRF')
plt.plot(fr, rf_list, label = 'RF')
plt.title('Noise in the labels')
plt.ylabel('Accuray')
plt.xlabel('fraction of missclassified')
plt.grid()
plt.legend()

# Noise in the features (simple case)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Ns = list(range(10))

prf_list = []
rf_list = []

for ns in Ns:
    prf = PRF()

    dx = inject_noise1(np.append(X_train, X_test, axis=0), ns)

    dx_train = dx[:X_train.shape[0]]
    dx_test = dx[X_train.shape[0]:]

    X_train = main_inject_noise(X_train, dx_train)
    X_test = main_inject_noise(X_test, dx_test)

    prf.fit(X_train, y_train, dX=dx_train)
    prf_predict = prf.predict(X_test, dX=dx_test)

    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(X_train, y_train)
    rf_predict = rf.predict(X_test)

    prf_score = accuracy_score(y_test, prf_predict)
    rf_score = accuracy_score(y_test, rf_predict)

    prf_list.append(prf_score)
    rf_list.append(rf_score)

plt.figure()
plt.plot(Ns, prf_list, label='PRF')
plt.plot(Ns, rf_list, label='RF')
plt.title('Noise in the features (simple case)')
plt.ylabel('Accuray')
plt.xlabel('Ns')
plt.grid()
plt.legend()

# Noise in the features (complex case)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Ns = list(range(10))
prf_list = []
rf_list = []

for ns in Ns:
    prf = PRF()

    dx = inject_noise2(np.append(X_train, X_test, axis=0), ns)

    dx_train = dx[:X_train.shape[0]]
    dx_test = dx[X_train.shape[0]:]

    X_train = main_inject_noise(X_train, dx_train)
    X_test = main_inject_noise(X_test, dx_test)

    prf.fit(X_train, y_train, dX=dx_train)
    prf_predict = prf.predict(X_test, dX=dx_test)

    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(X_train, y_train)
    rf_predict = rf.predict(X_test)

    prf_score = accuracy_score(y_test, prf_predict)
    rf_score = accuracy_score(y_test, rf_predict)

    prf_list.append(prf_score)
    rf_list.append(rf_score)

plt.figure()
plt.plot(Ns, prf_list, label='PRF')
plt.plot(Ns, rf_list, label='RF')
plt.title('Noise in the features (complex case)')
plt.ylabel('Accuray')
plt.xlabel('Ns')
plt.grid()
plt.legend()

# Different noise characteristics in the training and the test sets


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Ns = list(range(10))
prf_list = []
rf_list = []

for ns in Ns:
    prf = PRF()

    dx_train, dx_test = inject_noise3(X_train, X_test, ns)

    X_train = main_inject_noise(X_train, dx_train)
    X_test = main_inject_noise(X_test, dx_test)

    prf.fit(X_train, y_train, dX=dx_train)
    prf_predict = prf.predict(X_test, dX=dx_test)

    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(X_train, y_train)
    rf_predict = rf.predict(X_test)

    prf_score = accuracy_score(y_test, prf_predict)
    rf_score = accuracy_score(y_test, rf_predict)

    prf_list.append(prf_score)
    rf_list.append(rf_score)

plt.figure()
plt.plot(Ns, prf_list, label='PRF')
plt.plot(Ns, rf_list, label='RF')
plt.title('Different noise characteristics in the training and the test sets')
plt.ylabel('Accuray')
plt.xlabel('Ns')
plt.grid()
plt.legend()
plt.show()


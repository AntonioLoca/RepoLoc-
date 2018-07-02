from __future__ import print_function
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import csv


def centralize(D):
    mean = D.mean(axis=0)
    D_centr = D - mean
    return D_centr

def normalize(D):
    std = D.std(axis=0)
    std[std == 0] = 1
    D_centr = centralize(D)
    D_norm = D_centr / std
    return D_norm

def csvread(filename, delimiter='\t'):
    f = open(filename, 'r')
    reader = csv.reader(f, delimiter=delimiter)
    ncol = len(next(reader))
    nfeat = ncol - 1
    f.seek(0)
    x = np.zeros(nfeat)
    X = np.empty((0, nfeat))

    y = []
    for row in reader:
        for j in range(nfeat):
            x[j] = float(row[j])

        X = np.append(X, [x], axis=0)
        label = row[nfeat]
        y.append(label)

    lb = LabelBinarizer()
    Y = lb.fit_transform(y)
    classname = lb.classes_

    le = LabelEncoder()
    ynum = le.fit_transform(y)

    return X, ynum


def read_arq(A):
    filename = A
    delimiter = '\t'
    X1, ynum = csvread(filename=filename, delimiter=delimiter)
    X1 = normalize(X1)
    std = X1.std(axis=0)

    return X1, ynum



X_train, Y_train = read_arq('treino.csv')

X_test, Y_test = read_arq('teste.csv')



seed = 7
np.random.seed(seed)
model = Sequential()
model.add(Dense(20, activation="relu", kernel_initializer="uniform", input_dim=53))
model.add(Dense(12, activation="relu", kernel_initializer="uniform"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, nb_epoch=1000, batch_size=128, verbose=1)

scores = model.evaluate(X_test, Y_test)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


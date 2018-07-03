from __future__ import print_function
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from keras.layers import Dense
import numpy as np
import csv
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
import random

# Funcao para centralizar os dados.

def center(mtx):
    media = mtx.mean(axis=0)
    Mtx_centr = mtx - media
    return Mtx_centr

# Funcao para normalizar os dados.

def normalize(MTX):
    normalizado = MTX.std(axis=0)
    normalizado[normalizado == 0] = 1
    MT_centr = center(MTX)
    MT_normal = MT_centr / normalizado
    return MT_normal

# Formatacao do arquivo CSV.

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

# Leitura do arquivo CSV.

def read_arq(A):
    filename = A
    delimiter = '\t'
    X1, ynum = csvread(filename=filename, delimiter=delimiter)
    X1 = normalize(X1)
    std = X1.std(axis=0)

    return X1, ynum


X_train, Y_train = read_arq('treino.csv') # Arquivo contendo os estados normais e com falhas para treino.

X_test, Y_test = read_arq('teste.csv') # Arquivo contendo os estados normais e com falhas para teste.


# Embaralhamento das caracteristicas.

newMatrix = np.zeros((X_train.shape))
cols =(np.arange(X_train.shape[1]))
random.shuffle(newMatrix)

for i in range(len(X_train)):
    for j in range(len(X_train[0])):
        newMatrix[i][j] = X_train[i][cols[j]]


# Rede neural multicamada.

seed = 7

np.random.seed(seed)

model = Sequential()

model.add(Dense(12, activation="relu", kernel_initializer="uniform", input_dim=53))

model.add(Dense(8, activation="relu", kernel_initializer="uniform"))

model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=600, batch_size=128, verbose=1)

scores = model.evaluate(X_test, Y_test)


# Grafico e modelo da estrutura da rede.

plot_model(model)

plot_model(model, to_file = 'model_plot.png', show_shapes = True, show_layer_names = True)

# Desempenho de aprendizagem da rede.

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


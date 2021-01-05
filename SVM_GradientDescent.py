from math import exp
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_shuffled = pd.read_csv('heart_modified.csv')

# data_shuffled = data.sample(frac=1).reset_index(drop=True)
#
# data_shuffled.to_csv('heart_modified.csv')

age = data_shuffled.iloc[:, 0]
sex = data_shuffled.iloc[:, 1]
cp = data_shuffled.iloc[:, 2]
trestbps = data_shuffled.iloc[:, 3]
chol = data_shuffled.iloc[:, 4]
fbs = data_shuffled.iloc[:, 5]
restecg = data_shuffled.iloc[:, 6]
thalach = data_shuffled.iloc[:, 7]
exang = data_shuffled.iloc[:, 8]
oldpeak = data_shuffled.iloc[:, 9]
slope = data_shuffled.iloc[:, 10]
ca = data_shuffled.iloc[:, 11]
thal = data_shuffled.iloc[:, 12]

target = data_shuffled.iloc[:, 13]

# Train and test data

age_train = age[:257]
age_test = age[257:]

sex_train = sex[:257]
sex_test = sex[257:]

cp_train = cp[:257]
cp_test = cp[257:]

trestbps_train = trestbps[:257]
trestbps_test = trestbps[257:]

chol_train = chol[:257]
chol_test = chol[257:]

fbs_train = fbs[:257]
fbs_test = fbs[257:]

restecg_train = restecg[:257]
restecg_test = restecg[257:]

thalach_train = thalach[:257]
thalach_test = thalach[257:]

exang_train = exang[:257]
exang_test = exang[257:]

oldpeak_train = oldpeak[:257]
oldpeak_test = oldpeak[257:]

slope_train = slope[:257]
slope_test = slope[257:]

ca_train = ca[:257]
ca_test = ca[257:]

thal_train = thal[:257]
thal_test = thal[257:]

target_train = target[:257]
target_test = target[257:]



color = data_shuffled['target'].apply(lambda x: 'green' if x == 0 else 'red')
data_shuffled.plot(kind='scatter', x='thal', y='cp', c=color)

plt.show()

target_train = np.where(target_train == 0, -1, 1)

target_test = np.where(target_test == 0, -1, 1)

X_train = []
X_test = []

for f1, f2 in zip(thal_test, cp_test):
    X_test.append([f1, f2])
for f1, f2 in zip(thal_train, cp_train):
    X_train.append([f1, f2])

weight = np.zeros(2)

# b = 1   accuracy =  0.8260869565217391
# b = 0   accuracy =  0.8043478260869565

b = 1

epochs = 1000

# learning_rate = 0.1    accuracy =  0.43478260869565216
# learning_rate = 0.01   accuracy =  0.8260869565217391
# learning_rate = 0.001   accuracy =  0.8260869565217391
# learning_rate = 0.000001    accuracy =  0.43478260869565216

learning_rate = 0.01

lambdaa = 1 / epochs

# Fitting

for _ in range(epochs):

    for ind, x_i in enumerate(X_train):  # Updating weights
        if target_train[ind] * (np.dot(x_i, weight) + b) >= 1:
            weight = weight - learning_rate * (2 * lambdaa * weight)
        else:
            weight = weight + learning_rate * (np.dot(target_train[ind], x_i) - 2 * lambdaa * weight)

# Prediction
target_predicted = []

for x_i in X_test:
    target_predicted.append(np.dot(weight, x_i) + b)

# Approximation

for i in range(len(target_predicted)):

    if 0 < target_predicted[i] < 1:
        target_predicted[i] = 1
    elif 0 > target_predicted[i] > -1:
        target_predicted[i] = -1

# Calculating accuracy

acc = 0
for i in range(len(target_predicted)):
    if target_test[i] * target_predicted[i] >= 1:
        acc += 1
print("accuracy = ", acc / len(target_test))

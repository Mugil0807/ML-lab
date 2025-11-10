# Optimization of Fault Detection in Aquaponics System
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split as split, cross_val_score as cv_score
from pyswarm import pso

data = pd.read_excel("C:\\Users\\Rakshan D\\OneDrive - Amrita vishwa vidyapeetham\\SEM-III\\OT\\Cleansed pond dataset\\Realtime-env2_withfaults_4.xlsx")
features, target = data.drop(columns=['Fault']), data['Fault']
trainX, testX, trainY, testY = split(features, target, test_size=0.2, random_state=42)

mDefault = RF(random_state=42)
mDefault.fit(trainX, trainY)
baseline_acc = cv_score(mDefault, trainX, trainY, cv=2, scoring='accuracy').mean()
print(f"Default Accuracy is: {baseline_acc:.4f}")

def opt_fitness(param_set):
    trees, max_d = int(param_set[0]), int(param_set[1])
    rf_model = RF(nEstimators=trees, max_depth=max_d, random_state=42)
    accuracy = cv_score(rf_model, trainX, trainY, cv=2, scoring='accuracy').mean()
    return -accuracy

bounds_range = ([10, 1], [200, 20])
opt_params, _ = pso(opt_fitness, *bounds_range, swarmsize=10, maxiter=10)

opt_rf = RF(nEstimators=int(opt_params[0]), max_depth=int(opt_params[1]), random_state=42)
opt_rf.fit(trainX, trainY)
opt_acc = cv_score(opt_rf, trainX, trainY, cv=3, scoring='accuracy').mean()
print(f"Optimized Accuracy is: {opt_acc:.4f}")

acc_gain = (opt_acc - baseline_acc) * 100
print(f"Accuracy Improvement is : {acc_gain:.2f}%")

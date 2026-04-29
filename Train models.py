# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 16:20:47 2025

@author: Agrima Agarwal
"""

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score



# ---------------------------
#  Model functions
# ---------------------------

def PCA_model(X_train, X_test, y_train):
    pca = PCA(n_components=1, random_state=42)
    pca.fit(X_train)
    X_test_trans = pca.transform(X_test)
    return X_test_trans[:, 0]


def LDA_model(X_train, X_test, y_train):
    lda = LDA(n_components=1,priors=[0.5,0.5])   
    lda.fit(X_train, y_train)
    # X_test_trans = lda.transform(X_test)
    # return X_test_trans[:, 0]
    return lda.predict_proba(X_test)[:,1]


def logistic_regressor_model(X_train, X_test, y_train):
    clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=5000)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def decisiontree_model(X_train, X_test, y_train):
    clf = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def randomforest_model(X_train, X_test, y_train):
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', 
                                 random_state=42)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def kNN_model(X_train, X_test, y_train):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def SingleLayerPerceptron_model(X_train, X_test, y_train):
    clf = Perceptron(random_state=42)
    clf.fit(X_train, y_train)
    return clf.decision_function(X_test)


def MultiLayerPerceptron_model(X_train, X_test, y_train):
    clf = MLPClassifier(hidden_layer_sizes=(6,), learning_rate_init=0.1,
                        momentum=0.1, max_iter=200000, random_state=42)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


# ---------------------------
#  Settings
# ---------------------------

test_size = 0.3
n_bootstraps = 50
rng = np.random.RandomState(42)   # GLOBAL SEED


# ---------------------------
#  Load data
# ---------------------------

# data = np.load('outputs/Xy_dry-skin-type.npz')
# data = np.load('outputs/Xy_dry-skin-no-moist.npz')
# data = np.load('outputs/Xy_dry-skin-10-moist.npz')
# data = np.load('outputs/Xy_dry-skin-20-moist.npz')
# data = np.load('outputs/Xy_dry-skin-type.npz')
data = np.load('outputs/Xy_skin-cancer.npz')

X = data['X']
y = data['y']

# patient IDs: 1..n, 1..n
n_patients = len(y) // 2
unique_patients = np.arange(1, n_patients + 1)
patient_ids = np.concatenate([unique_patients, unique_patients])


# ---------------------------
#  Model list
# ---------------------------

classifiers = [
    ('PC1', PCA_model),
    ('LDA', LDA_model),
    ('logistic regression', logistic_regressor_model),
    ('logistic regression on PCA', 'logreg_PCA'),
    ('knn', kNN_model),
    ('knn on PCA', 'knn_PCA'),
    ('decision tree', decisiontree_model),
    ('random forest', randomforest_model),
    ('SLP', SingleLayerPerceptron_model),
    ('MLP', MultiLayerPerceptron_model)
]


# ---------------------------
#  Evaluation
# ---------------------------

roc_auc_values = {}

for name, model_fn in classifiers:
    print(f"Running: {name}")
    boot_aucs = []

    for i in range(n_bootstraps):
        # ---- fixed seed for reproducible patient split ----
        train_patients, test_patients = train_test_split(
            unique_patients, test_size=test_size, random_state=i
        )

        train_mask = np.isin(patient_ids, train_patients)
        test_mask  = np.isin(patient_ids, test_patients)

        # ---- bootstrap whole patients ----
        boot_patients = rng.choice(train_patients, size=len(train_patients), replace=True)
        bootstrap_indices = np.concatenate(
            [np.where(patient_ids == p)[0] for p in boot_patients]
        )

        X_train = X[bootstrap_indices]
        y_train = y[bootstrap_indices]
        X_test = X[test_mask]
        y_test = y[test_mask]

        # ---- PCA-based models ----
        if model_fn == 'logreg_PCA':
            pca = PCA(n_components=2, random_state=42)
            X_train_p = pca.fit_transform(X_train)
            X_test_p = pca.transform(X_test)
            y_score = logistic_regressor_model(X_train_p, X_test_p, y_train)

        elif model_fn == 'knn_PCA':
            pca = PCA(n_components=2, random_state=42)
            X_train_p = pca.fit_transform(X_train)
            X_test_p = pca.transform(X_test)
            y_score = kNN_model(X_train_p, X_test_p, y_train)

        else:
            y_score = model_fn(X_train, X_test, y_train)

        auc_val = roc_auc_score(y_test, y_score)
        boot_aucs.append(auc_val)

    roc_auc_values[name] = np.array(boot_aucs)


# ---------------------------
#  Final output
# ---------------------------

for name in roc_auc_values:
    mean_auc = roc_auc_values[name].mean()
    std_auc = roc_auc_values[name].std()
    print(f"{name}, {mean_auc*100:.2f}, {std_auc*100:.2f}")



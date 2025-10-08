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




def PCA_model(X_train,X_test,y_train):
    pca = PCA(n_components=20)
    pca.fit_transform(X_train)
    X_test_trans = pca.transform(X_test)
    return X_test_trans[:,0]

def LDA_model(X_train,X_test,y_train):
    lda = LDA(n_components=1,priors=[0.5,0.5])
    lda.fit_transform(X_train, y_train)
    X_test_trans = lda.transform(X_test)
    return X_test_trans[:,0]

def logistic_regressor_model(X_train,X_test,y_train):
    logistic_regressor = LogisticRegression(class_weight='balanced')
    logistic_regressor.fit(X_train, y_train)
    X_test_trans = logistic_regressor.predict_proba(X_test)[:, 1]
    return X_test_trans

def decisiontree_model(X_train,X_test,y_train):
    decision_tree = DecisionTreeClassifier(class_weight='balanced')
    decision_tree.fit(X_train, y_train)
    X_test_trans = decision_tree.predict_proba(X_test)[:, 1]
    return X_test_trans

def randomforest_model(X_train,X_test,y_train):
    random_forest = RandomForestClassifier(n_estimators=100,class_weight='balanced')
    random_forest.fit(X_train, y_train)
    X_test_trans = random_forest.predict_proba(X_test)[:, 1]
    return X_test_trans

def kNN_model(X_train,X_test,y_train):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    X_test_trans = knn.predict_proba(X_test)[:, 1]
    return X_test_trans

def SingleLayerPerceptron_model(X_train,X_test,y_train):
    perceptron  = Perceptron()
    perceptron.fit(X_train, y_train)
    X_test_trans = perceptron.decision_function(X_test)
    return X_test_trans

def MultiLayerPerceptron_model(X_train,X_test,y_train):
    mlp = MLPClassifier(hidden_layer_sizes=(6,), learning_rate_init=0.1, momentum=0.1, max_iter=200000)
    # mlp = MLPClassifier(hidden_layer_sizes=(6,), learning_rate_init=0.001, alpha=0.0001, max_iter=1000)

    mlp.fit(X_train, y_train)
    X_test_trans = mlp.predict_proba(X_test)[:, 1]
    return X_test_trans

test_size = 0.3
n_bootstraps = 50
rng = np.random.RandomState(42)  # Seed for reproducibility

#%% Load X and y

data = np.load('Xy_dry-skin-type.npz')
X = data['X']
y = data['y']

# Separate positive and negative classes
positive_indices = np.where(y == 1)[0]
negative_indices = np.where(y == 0)[0]

#%% Bootstrap runs for all models and corresponding AUROC values

bootstrap_auroc = []
roc_auc_values = {}


classifiers=[('PC1',PCA_model),
             ('LDA',LDA_model),
             ('logistic regression',logistic_regressor_model),
             ('logistic regression on PCA','logistic_regressor_model_PCA'), 
             ('knn',kNN_model),
             ('knn on PCA','kNN_model_PCA'),
             ('decision tree', decisiontree_model),
             ('random forest',randomforest_model),
             ('SLP',SingleLayerPerceptron_model),
             ('MLP',MultiLayerPerceptron_model)]

for name,j in classifiers:
    bootstrap_auroc=[]
    for i in range(n_bootstraps):
        bootstrap_positive_indices = rng.choice(positive_indices, size=len(positive_indices), replace=True)
        bootstrap_negative_indices = rng.choice(negative_indices, size=len(negative_indices), replace=True)
        # Combine indices
        bootstrap_indices = np.concatenate((bootstrap_positive_indices, bootstrap_negative_indices))
        if j == 'logistic_regressor_model_PCA':
            pca = PCA(n_components=2)
            X_trans = pca.fit_transform(X)
            X_boot = X_trans[bootstrap_indices]
            y_boot = y[bootstrap_indices]
            X_train, X_test, y_train, y_test = train_test_split(X_boot, y_boot, test_size=test_size, stratify=y_boot,random_state=i)
            X_test_trans = logistic_regressor_model(X_train, X_test, y_train)   
        elif j == 'kNN_model_PCA':
            pca = PCA(n_components=2)
            X_trans = pca.fit_transform(X)
            X_boot = X_trans[bootstrap_indices]
            y_boot = y[bootstrap_indices]
            X_train, X_test, y_train, y_test = train_test_split(X_boot, y_boot, test_size=test_size, stratify=y_boot,random_state=i)
            X_test_trans = kNN_model(X_train, X_test, y_train)   
        else:
            X_boot = X[bootstrap_indices]
            y_boot = y[bootstrap_indices]
            X_train, X_test, y_train, y_test = train_test_split(X_boot, y_boot, test_size=test_size, stratify=y_boot,random_state=i)
            X_test_trans = j(X_train, X_test, y_train)    
        score = roc_auc_score(y_test, X_test_trans)
        bootstrap_auroc.append(score)
    roc_auc_values[name] = np.array(bootstrap_auroc)

for i in roc_auc_values:
    print('%s, %s, %s' % (i,roc_auc_values[i].mean(),roc_auc_values[i].std()))


#%% Statistics for random forest

n_bootstraps = 50
fpr_grid = np.linspace(0, 1, 100)
tpr_bootstraps = []
bootstrap_auroc = []
sensitivities = []
specificities = []
accuracies = []
thresholds = []
for i in range(n_bootstraps):
    bootstrap_positive_indices = rng.choice(positive_indices, size=len(positive_indices), replace=True)
    bootstrap_negative_indices = rng.choice(negative_indices, size=len(negative_indices), replace=True)
    # Combine indices
    bootstrap_indices = np.concatenate((bootstrap_positive_indices, bootstrap_negative_indices))
    X_boot = X[bootstrap_indices]
    y_boot = y[bootstrap_indices]
    X_train, X_test, y_train, y_test = train_test_split(X_boot, y_boot, test_size=test_size, stratify=y_boot,random_state=i)
    X_test_trans = randomforest_model(X_train, X_test, y_train) 
    y_true = y_test
    y_score = X_test_trans
    fpr, tpr, thresh = roc_curve(y_test, X_test_trans)
    score = roc_auc_score(y_test, X_test_trans)
    bootstrap_auroc.append(score)
    tpr_interp = np.interp(fpr_grid, fpr, tpr)
    tpr_interp[0] = 0.0
    tpr_bootstraps.append(tpr_interp)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresh[best_idx]
    thresholds.append(best_thresh)
    # Binarize predictions
    y_pred = (y_score >= best_thresh).astype(int)
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Metrics
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_true, y_pred)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    accuracies.append(accuracy)

# Convert to arrays
sensitivities = np.array(sensitivities)
specificities = np.array(specificities)
accuracies = np.array(accuracies)
thresholds = np.array(thresholds)
print(bootstrap_auroc)
# np.savez('AUROC_dry-skin-type', auroc = bootstrap_auroc)
# Compute means and 95% confidence intervals
def mean_ci(arr):
    return np.mean(arr), np.percentile(arr, 2.5), np.percentile(arr, 97.5)

sens_mean, sens_low, sens_high = mean_ci(sensitivities)
spec_mean, spec_low, spec_high = mean_ci(specificities)
acc_mean, acc_low, acc_high = mean_ci(accuracies)
thresh_mean, thresh_low, thresh_high = mean_ci(thresholds)

# Display results
print(f"Sensitivity: {sens_mean:.3f} (95% CI: {sens_low:.3f} – {sens_high:.3f})")
print(f"Specificity: {spec_mean:.3f} (95% CI: {spec_low:.3f} – {spec_high:.3f})")
print(f"Accuracy:    {acc_mean:.3f} (95% CI: {acc_low:.3f} – {acc_high:.3f})")
print(f"Threshold:   {thresh_mean:.3f} (95% CI: {thresh_low:.3f} – {thresh_high:.3f})")

tpr_bootstraps = np.array(tpr_bootstraps)
mean_tpr = tpr_bootstraps.mean(axis=0)
std_tpr = tpr_bootstraps.std(axis=0)
ci_lower = np.maximum(mean_tpr - 1.96 * std_tpr / np.sqrt(n_bootstraps), 0)
ci_upper = np.minimum(mean_tpr + 1.96 * std_tpr / np.sqrt(n_bootstraps), 1)

# Compute AUCs if desired
aucs = [auc(fpr_grid, tpr) for tpr in tpr_bootstraps]
mean_auc = np.mean(aucs)
std_auc = np.std(aucs)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_grid, mean_tpr, color='blue', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
plt.fill_between(fpr_grid, ci_lower, ci_upper, color='blue', alpha=0.2, label='95% confidence interval')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with 95% Confidence Interval')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()

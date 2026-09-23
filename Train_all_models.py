# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


N_REPEATS = 50
OUTER_TEST_SIZE = 0.30
INNER_VALIDATION_SIZE = 0.20   # of the 70% development pool
SEED_BASE = 1000

OUTPUT_DIR = Path("outputs/training_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

task_no = 2

TASKS = [
    ("Dry vs Healthy - no moisturizer", "outputs/Xy_same/Xy_dry-skin-no-moist.npz", "paired_regions"),
    ("Dry vs Healthy - 10 min moisturizer", "outputs/Xy_same/Xy_dry-skin-10-moist.npz", "paired_regions"),
    ("Dry vs Healthy - 20 min moisturizer", "outputs/Xy_same/Xy_dry-skin-20-moist.npz", "paired_regions"),
    ("Eczema vs Psoriasis", "outputs/Xy/Xy_dry-skin-type.npz", "single_label"),
    ("Skin Cancer vs Healthy", "outputs/Xy/Xy_skin-cancer.npz", "paired_regions"),
]


PARAMETER_GRIDS = {
    "PC1": [{}],
    "LDA": [{}],

    "Logistic Regression": [
        {"C": c}
        for c in [0.001, 0.01, 0.1]
    ],

    "Logistic Regression + PCA": [
        {"n_components": n, "C": c}
        for n in [2, 3, 5]
        for c in [0.001, 0.01, 0.1]
    ],

    "kNN": [
        {"n_neighbors": k, "weights": w}
        for k in [5, 7, 9]
        for w in ["uniform", "distance"]
    ],

    "kNN + PCA": [
        {"n_components": n, "n_neighbors": k, "weights": w}
        for n in [2, 3, 5]
        for k in [5, 7, 9]
        for w in ["uniform", "distance"]
    ],

    "Decision Tree": [
        {"max_depth": depth, "min_samples_leaf": leaf}
        for depth in [3, 10]
        for leaf in [2, 4, 6]
    ],

    "Random Forest": [
        {
            "n_estimators": n_trees,
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "max_features": max_features,
        }
        for n_trees in [100, 500]
        for depth in [3, 10]
        for leaf in [1, 2, 4]
        for max_features in ["sqrt", "log2"]
    ],

    "SLP": [
        {"alpha": alpha}
        for alpha in [1e-4, 1e-3, 1e-2]
    ],

    "MLP": [
        {
            "hidden_layer_sizes": hidden,
            "alpha": alpha,
            "learning_rate_init": 0.01,
        }
        for hidden in [(4,), (16,)]
        for alpha in [1e-4, 1e-3, 1e-2]
    ],
}

CLASSICAL_MODELS = list(PARAMETER_GRIDS)

CNN_LR = 1e-3
CNN_WEIGHT_DECAY = 1e-4
CNN_KERNEL_SIZE = 3
CNN_DROPOUT = 0.0
CNN_BATCH_SIZE = 8
CNN_MAX_EPOCHS = 50
CNN_PATIENCE = 6
CNN_MIN_DELTA = 1e-4


# Set NumPy and PyTorch random seeds.
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Return mean and standard deviation while ignoring NaNs.
def mean_sd(values):
    a = np.asarray(values, dtype=float)
    if np.all(np.isnan(a)):
        return np.nan, np.nan
    return float(np.nanmean(a)), float(np.nanstd(a))


# Create patient IDs for patient-independent splitting.
def make_patient_ids(y, structure):
    if structure == "single_label":
        unique_patients = np.arange(len(y))
        patient_ids = unique_patients.copy()
    elif structure == "paired_regions":
        if len(y) % 2:
            raise ValueError("paired_regions requires an even number of samples")
        n_patients = len(y) // 2
        unique_patients = np.arange(n_patients)
        patient_ids = np.concatenate([unique_patients, unique_patients])
    else:
        raise ValueError("structure must be 'single_label' or 'paired_regions'")
    return unique_patients, patient_ids


# Create train, validation and test patient splits.
def split_patients(y, unique_patients, structure, repeat_i):
    if structure == "single_label":
        train_pool, test_patients = train_test_split(
            unique_patients,
            test_size=OUTER_TEST_SIZE,
            random_state=repeat_i,
            stratify=y,
        )
        train_patients, val_patients = train_test_split(
            train_pool,
            test_size=INNER_VALIDATION_SIZE,
            random_state=repeat_i,
            stratify=y[train_pool],
        )
    else:
        train_pool, test_patients = train_test_split(
            unique_patients,
            test_size=OUTER_TEST_SIZE,
            random_state=repeat_i,
        )
        train_patients, val_patients = train_test_split(
            train_pool,
            test_size=INNER_VALIDATION_SIZE,
            random_state=repeat_i,
        )
    return train_patients, val_patients, test_patients


# Return ROC points with finite thresholds.
def finite_roc(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    keep = np.isfinite(thresholds)
    return fpr[keep], tpr[keep], thresholds[keep]


# Select the threshold that maximises Youden's J statistic.
def youden_threshold(y_true, scores):
    fpr, tpr, thresholds = finite_roc(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    return float(thresholds[np.argmax(tpr - fpr)])


# Calculate sensitivity, specificity and accuracy at a threshold.
def metrics_at_threshold(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    accuracy = accuracy_score(y_true, pred)
    return float(sensitivity), float(specificity), float(accuracy)


# Find maximum sensitivity subject to minimum specificity.
def sensitivity_at_min_specificity(y_true, scores, min_specificity=0.80):
    fpr, tpr, thresholds = finite_roc(y_true, scores)
    specificity = 1.0 - fpr
    valid = np.where(specificity >= min_specificity)[0]
    if len(valid) == 0:
        return np.nan, np.nan, np.nan
    best_sens = np.max(tpr[valid])
    tied = valid[np.where(np.isclose(tpr[valid], best_sens))[0]]
    idx = tied[np.argmax(specificity[tied])]
    return float(tpr[idx]), float(specificity[idx]), float(thresholds[idx])


# Find maximum specificity subject to minimum sensitivity.
def specificity_at_min_sensitivity(y_true, scores, min_sensitivity=0.80):
    fpr, tpr, thresholds = finite_roc(y_true, scores)
    specificity = 1.0 - fpr
    valid = np.where(tpr >= min_sensitivity)[0]
    if len(valid) == 0:
        return np.nan, np.nan, np.nan
    best_spec = np.max(specificity[valid])
    tied = valid[np.where(np.isclose(specificity[valid], best_spec))[0]]
    idx = tied[np.argmax(tpr[tied])]
    return float(tpr[idx]), float(specificity[idx]), float(thresholds[idx])


# Fit a classical model and return evaluation scores.
def classical_fit_score(name, params, X_train, y_train, X_eval):

    if name == "PC1":
        pca = PCA(n_components=1, random_state=42)
        tr = pca.fit_transform(X_train)[:, 0]
        ev = pca.transform(X_eval)[:, 0]
        if np.mean(tr[y_train == 1]) < np.mean(tr[y_train == 0]):
            ev = -ev
        return ev, np.nan, pca

    if name == "LDA":
        clf = LDA(n_components=1, priors=[0.5, 0.5])
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_eval)[:, 1], np.nan, clf

    if name == "Logistic Regression":
        clf = LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=5000,
            C=params["C"],
        )
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_eval)[:, 1], int(clf.n_iter_[0]), clf

    if name == "Logistic Regression + PCA":
        pca = PCA(n_components=params["n_components"], random_state=42)
        Xtr = pca.fit_transform(X_train)
        Xev = pca.transform(X_eval)
        clf = LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=5000,
            C=params["C"],
        )
        clf.fit(Xtr, y_train)
        return clf.predict_proba(Xev)[:, 1], int(clf.n_iter_[0]), (pca, clf)

    if name == "kNN":
        clf = KNeighborsClassifier(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
        )
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_eval)[:, 1], np.nan, clf

    if name == "kNN + PCA":
        pca = PCA(n_components=params["n_components"], random_state=42)
        Xtr = pca.fit_transform(X_train)
        Xev = pca.transform(X_eval)
        clf = KNeighborsClassifier(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
        )
        clf.fit(Xtr, y_train)
        return clf.predict_proba(Xev)[:, 1], np.nan, (pca, clf)

    if name == "Decision Tree":
        clf = DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
        )
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_eval)[:, 1], np.nan, clf

    if name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_eval)[:, 1], np.nan, clf

    if name == "SLP":
        clf = Perceptron(
            penalty="l2",
            alpha=params["alpha"],
            max_iter=1000,
            tol=1e-3,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        return clf.decision_function(X_eval), int(clf.n_iter_), clf

    if name == "MLP":
        clf = MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            alpha=params["alpha"],
            learning_rate_init=0.01,
            max_iter=5000,
            tol=1e-4,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_eval)[:, 1], int(clf.n_iter_), clf

    raise ValueError(name)


# Two-block 1D CNN used for the final classifier.
class TwoBlockCNN1D(nn.Module):
    # Build the two-block CNN.
    def __init__(self, input_channels=229, input_length=400):
        super().__init__()
        p = CNN_KERNEL_SIZE // 2
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, CNN_KERNEL_SIZE, padding=p),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, CNN_KERNEL_SIZE, padding=p),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_length)
            flattened_features = self.features(dummy).numel()

        self.dropout = nn.Dropout(CNN_DROPOUT) if CNN_DROPOUT > 0 else nn.Identity()
        self.fc = nn.Linear(flattened_features, 2)

    # Run the forward pass.
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        return self.fc(x)


# Track validation loss for early stopping.
class EarlyStopping:
    # Initialize early-stopping parameters and state.
    def __init__(self, patience=CNN_PATIENCE, min_delta=CNN_MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.best_state = None
        self.best_epoch = np.nan
        self.counter = 0
        self.stop = False

    # Update the early-stopping state.
    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# Reshape flattened measurements to [N, 229, 400] for the CNN.
def prepare_cnn_X(X, y):
    return torch.tensor(
        X.reshape(len(y), 400, 229, order="F"),
        dtype=torch.float32,
    ).permute(0, 2, 1)


# Create a PyTorch data loader.
def make_loader(X, y=None, shuffle=False, seed=0):
    if y is None:
        ds = TensorDataset(X)
    else:
        ds = TensorDataset(X, torch.tensor(y, dtype=torch.long))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=CNN_BATCH_SIZE,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


# Train the CNN with validation-based early stopping.
def train_cnn(X_train, y_train, X_val, y_val, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoBlockCNN1D(input_channels=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CNN_LR, weight_decay=CNN_WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(X_train, y_train, shuffle=True, seed=seed)
    val_loader = make_loader(X_val, y_val, shuffle=False, seed=seed)
    early = EarlyStopping()
    epochs_run = 0

    for epoch in range(1, CNN_MAX_EPOCHS + 1):
        epochs_run = epoch
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                total_loss += loss.item() * len(yb)
                n += len(yb)
        val_loss = total_loss / n
        early.step(val_loss, model, epoch)
        if early.stop:
            break

    if early.best_state is not None:
        model.load_state_dict(early.best_state)

    return model, device, epochs_run, int(early.best_epoch), float(early.best_loss)


# Return class-1 softmax scores from the CNN.
def cnn_score(model, device, X):
    model.eval()
    scores = []
    with torch.no_grad():
        for (xb,) in make_loader(X):
            probs = torch.softmax(model(xb.to(device)), dim=1)[:, 1]
            scores.extend(probs.cpu().numpy())
    return np.asarray(scores)


# Apply Holm correction to multiple p-values.
def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    out_sorted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        out_sorted[rank] = min(running, 1.0)
    out = np.empty(m)
    for rank, idx in enumerate(order):
        out[idx] = out_sorted[rank]
    return out


# Compare the best-AUROC model with the remaining models.
def best_vs_others_holm(result_store):
    best = max(result_store, key=lambda m: np.mean(result_store[m]["test_auc"]))
    best_auc = np.asarray(result_store[best]["test_auc"])
    rows, raw = [], []
    for other in result_store:
        if other == best:
            continue
        other_auc = np.asarray(result_store[other]["test_auc"])
        d = best_auc - other_auc
        if np.allclose(d, 0):
            W, p = 0.0, 1.0
        else:
            res = wilcoxon(best_auc, other_auc, alternative="two-sided")
            W, p = float(res.statistic), float(res.pvalue)
        rows.append({
            "best_model": best,
            "other_model": other,
            "W": W,
            "p_raw": p,
            "mean_delta_auc": float(np.mean(d)),
        })
        raw.append(p)
    adjusted = holm_adjust(raw)
    for row, p_holm in zip(rows, adjusted):
        row["p_holm"] = float(p_holm)
    return best, rows


summary_rows = []
repeat_rows = []
hyperparameter_rows = []
significance_rows = []
operating_rows = []

task_name, data_file, structure = TASKS[task_no]
print("\n" + "=" * 100)
print(task_name)
print("=" * 100)

data = np.load(data_file)
X = data["X"]
y = data["y"].astype(int)
unique_patients, patient_ids = make_patient_ids(y, structure)
X_cnn = prepare_cnn_X(X, y)

result_store = {
    name: {
        "test_auc": [], "test_auprc": [], "accuracy": [], "sensitivity": [],
        "specificity": [], "threshold": [], "iterations": [],
        "best_epoch": [], "test_scores": [], "test_y": [],
    }
    for name in CLASSICAL_MODELS + ["CNN"]
}

candidate_val_auc = {
    name: {repr(p): [] for p in PARAMETER_GRIDS[name]}
    for name in CLASSICAL_MODELS
}
selection_counts = {
    name: {repr(p): 0 for p in PARAMETER_GRIDS[name]}
    for name in CLASSICAL_MODELS
}

for repeat_i in range(N_REPEATS):
    print(f"{task_name}: repeat {repeat_i + 1}/{N_REPEATS}")

    train_patients, val_patients, test_patients = split_patients(
        y, unique_patients, structure, repeat_i
    )
    train_mask = np.isin(patient_ids, train_patients)
    val_mask = np.isin(patient_ids, val_patients)
    test_mask = np.isin(patient_ids, test_patients)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    for model_name in CLASSICAL_MODELS:
        best_params, best_val_auc = None, -np.inf

        for params in PARAMETER_GRIDS[model_name]:
            val_scores, _, _ = classical_fit_score(
                model_name, params, X_train, y_train, X_val
            )
            val_auc = roc_auc_score(y_val, val_scores)
            candidate_val_auc[model_name][repr(params)].append(val_auc)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_params = params

        selection_counts[model_name][repr(best_params)] += 1

        test_scores, iterations, _ = classical_fit_score(
            model_name, best_params, X_train, y_train, X_test
        )
        test_auc = roc_auc_score(y_test, test_scores)
        test_auprc = average_precision_score(y_test, test_scores)
        threshold = youden_threshold(y_test, test_scores)
        sens, spec, acc = metrics_at_threshold(y_test, test_scores, threshold)

        r = result_store[model_name]
        r["test_auc"].append(test_auc)
        r["test_auprc"].append(test_auprc)
        r["accuracy"].append(acc)
        r["sensitivity"].append(sens)
        r["specificity"].append(spec)
        r["threshold"].append(threshold)
        r["iterations"].append(iterations)
        r["best_epoch"].append(np.nan)
        r["test_scores"].append(test_scores.copy())
        r["test_y"].append(y_test.copy())

        repeat_rows.append({
            "task": task_name,
            "repeat": repeat_i,
            "model": model_name,
            "selected_params": repr(best_params),
            "selected_validation_auc": best_val_auc,
            "test_auc": test_auc,
            "test_auprc": test_auprc,
            "test_youden_threshold": threshold,
            "test_sensitivity": sens,
            "test_specificity": spec,
            "test_accuracy": acc,
            "iterations_to_stop": iterations,
            "cnn_best_epoch": np.nan,
            "cnn_epochs_run": np.nan,
        })

    model, device, epochs_run, best_epoch, best_val_loss = train_cnn(
        X_cnn[train_mask], y_train,
        X_cnn[val_mask], y_val,
        SEED_BASE + repeat_i,
    )
    test_scores = cnn_score(model, device, X_cnn[test_mask])
    test_auc = roc_auc_score(y_test, test_scores)
    test_auprc = average_precision_score(y_test, test_scores)
    threshold = youden_threshold(y_test, test_scores)
    sens, spec, acc = metrics_at_threshold(y_test, test_scores, threshold)

    r = result_store["CNN"]
    r["test_auc"].append(test_auc)
    r["test_auprc"].append(test_auprc)
    r["accuracy"].append(acc)
    r["sensitivity"].append(sens)
    r["specificity"].append(spec)
    r["threshold"].append(threshold)
    r["iterations"].append(epochs_run)
    r["best_epoch"].append(best_epoch)
    r["test_scores"].append(test_scores.copy())
    r["test_y"].append(y_test.copy())

    repeat_rows.append({
        "task": task_name,
        "repeat": repeat_i,
        "model": "CNN",
        "selected_params": repr({
            "architecture": "two_blocks",
            "lr": CNN_LR,
            "weight_decay": CNN_WEIGHT_DECAY,
            "kernel_size": CNN_KERNEL_SIZE,
            "dropout": CNN_DROPOUT,
        }),
        "selected_validation_auc": np.nan,
        "test_auc": test_auc,
        "test_auprc": test_auprc,
        "test_youden_threshold": threshold,
        "test_sensitivity": sens,
        "test_specificity": spec,
        "test_accuracy": acc,
        "iterations_to_stop": epochs_run,
        "cnn_best_epoch": best_epoch,
        "cnn_epochs_run": epochs_run,
        "cnn_best_validation_loss": best_val_loss,
    })

for model_name in CLASSICAL_MODELS:
    for params in PARAMETER_GRIDS[model_name]:
        vals = candidate_val_auc[model_name][repr(params)]
        m, s = mean_sd(vals)
        hyperparameter_rows.append({
            "task": task_name,
            "model": model_name,
            "parameters": repr(params),
            "validation_auc_mean": m,
            "validation_auc_sd": s,
            "selected_count": selection_counts[model_name][repr(params)],
        })

for model_name, r in result_store.items():
    auc_m, auc_s = mean_sd(r["test_auc"])
    auprc_m, auprc_s = mean_sd(r["test_auprc"])
    acc_m, acc_s = mean_sd(r["accuracy"])
    sens_m, sens_s = mean_sd(r["sensitivity"])
    spec_m, spec_s = mean_sd(r["specificity"])
    thr_m, thr_s = mean_sd(r["threshold"])
    it_m, it_s = mean_sd(r["iterations"])
    ep_m, ep_s = mean_sd(r["best_epoch"])

    summary_rows.append({
        "task": task_name,
        "model": model_name,
        "test_auc_mean": auc_m,
        "test_auc_sd": auc_s,
        "test_auprc_mean": auprc_m,
        "test_auprc_sd": auprc_s,
        "accuracy_mean": acc_m,
        "accuracy_sd": acc_s,
        "sensitivity_mean": sens_m,
        "sensitivity_sd": sens_s,
        "specificity_mean": spec_m,
        "specificity_sd": spec_s,
        "youden_threshold_mean": thr_m,
        "youden_threshold_sd": thr_s,
        "iterations_or_epochs_run_mean": it_m,
        "iterations_or_epochs_run_sd": it_s,
        "cnn_best_epoch_mean": ep_m,
        "cnn_best_epoch_sd": ep_s,
    })

best_model, sig_rows = best_vs_others_holm(result_store)
for row in sig_rows:
    row["task"] = task_name
    significance_rows.append(row)

best_result = result_store[best_model]
for repeat_i, (y_test, scores) in enumerate(
    zip(best_result["test_y"], best_result["test_scores"])
):
    youden_t = youden_threshold(y_test, scores)
    y_sens, y_spec, y_acc = metrics_at_threshold(y_test, scores, youden_t)

    sens80, achieved_spec80, thr_spec80 = sensitivity_at_min_specificity(
        y_test, scores, min_specificity=0.80
    )
    achieved_sens80, spec80, thr_sens80 = specificity_at_min_sensitivity(
        y_test, scores, min_sensitivity=0.80
    )

    operating_rows.append({
        "task": task_name,
        "best_model": best_model,
        "repeat": repeat_i,
        "test_auc": roc_auc_score(y_test, scores),
        "test_auprc": average_precision_score(y_test, scores),

        "youden_threshold": youden_t,
        "youden_sensitivity": y_sens,
        "youden_specificity": y_spec,
        "youden_accuracy": y_acc,

        "sensitivity_at_specificity_ge_80": sens80,
        "achieved_specificity_ge_80": achieved_spec80,
        "threshold_for_specificity_ge_80": thr_spec80,

        "specificity_at_sensitivity_ge_80": spec80,
        "achieved_sensitivity_ge_80": achieved_sens80,
        "threshold_for_sensitivity_ge_80": thr_sens80,
    })


# SAVE ROC DATA

roc_data_path = OUTPUT_DIR / (str(task_no) + " all_tasks_roc_data.npz")
roc_save = {
    "task_name": np.array(task_name),
    "n_repeats": np.array(N_REPEATS),
    "model_names": np.array(list(result_store.keys()), dtype=str),
}

for model_name, r in result_store.items():
    safe_name = model_name.replace(" ", "_").replace("+", "plus")
    for repeat_i, (y_test_saved, scores_saved) in enumerate(zip(r["test_y"], r["test_scores"])):
        roc_save[f"{safe_name}__repeat_{repeat_i:02d}__y_test"] = np.asarray(y_test_saved, dtype=int)
        roc_save[f"{safe_name}__repeat_{repeat_i:02d}__scores"] = np.asarray(scores_saved, dtype=float)

np.savez_compressed(roc_data_path, **roc_save)


# SAVE OUTPUTS

summary_df = pd.DataFrame(summary_rows)
repeat_df = pd.DataFrame(repeat_rows)
hyper_df = pd.DataFrame(hyperparameter_rows)
sig_df = pd.DataFrame(significance_rows)
operating_df = pd.DataFrame(operating_rows)

summary_path = OUTPUT_DIR / (str(task_no) + " all_tasks_model_summary.csv")
repeat_path = OUTPUT_DIR / (str(task_no) + " all_tasks_per_repeat_results.csv")
hyper_path = OUTPUT_DIR / (str(task_no) +  "all_tasks_hyperparameter_validation_results.csv")
sig_path = OUTPUT_DIR / (str(task_no) + " all_tasks_best_vs_others_holm.csv")
operating_path = OUTPUT_DIR / (str(task_no) + " all_tasks_best_model_operating_points.csv")
report_path = OUTPUT_DIR / (str(task_no) + " all_tasks_results.txt")

summary_df.to_csv(summary_path, index=False)
repeat_df.to_csv(repeat_path, index=False)
hyper_df.to_csv(hyper_path, index=False)
sig_df.to_csv(sig_path, index=False)
operating_df.to_csv(operating_path, index=False)

with report_path.open("w", encoding="utf-8") as f:
    f.write("ALL TASKS: CLASSICAL ML + 2-BLOCK CNN\n")
    f.write("=" * 100 + "\n\n")
    f.write("Split: 56% train / 14% validation / 30% test, patient-independent.\n")
    f.write("Validation selects classical hyperparameters and CNN early stopping.\n")
    f.write("Test Youden threshold is used for descriptive sensitivity/specificity/accuracy.\n")
    f.write("Special operating points use specificity >=80% or sensitivity >=80%.\n\n")

    f.write("MLP learning rate fixed at 0.01.\n")
    f.write("CNN architecture: two blocks.\n")
    f.write(f"CNN LR={CNN_LR}, weight_decay={CNN_WEIGHT_DECAY}, max_epochs={CNN_MAX_EPOCHS}.\n\n")

    f.write("CLASSICAL PARAMETER GRIDS\n")
    f.write("-" * 100 + "\n")
    f.write(json.dumps(PARAMETER_GRIDS, indent=2, default=str))
    f.write("\n\n")

    
    f.write("\n" + "=" * 100 + "\n")
    f.write(task_name + "\n")
    f.write("=" * 100 + "\n")

    sub = summary_df[summary_df.task == task_name]
    best_row = sub.loc[sub.test_auc_mean.idxmax()]

    for _, row in sub.iterrows():
        f.write(
            f"\n{row['model']}\n"
            f"  Test AUROC: {row['test_auc_mean']*100:.2f} ± {row['test_auc_sd']*100:.2f}%\n"
            f"  Test AUPRC: {row['test_auprc_mean']*100:.2f} ± {row['test_auprc_sd']*100:.2f}%\n"
            f"  Sensitivity (test Youden): {row['sensitivity_mean']*100:.2f} ± {row['sensitivity_sd']*100:.2f}%\n"
            f"  Specificity (test Youden): {row['specificity_mean']*100:.2f} ± {row['specificity_sd']*100:.2f}%\n"
            f"  Accuracy (test Youden): {row['accuracy_mean']*100:.2f} ± {row['accuracy_sd']*100:.2f}%\n"
            f"  Test Youden threshold: {row['youden_threshold_mean']:.4f} ± {row['youden_threshold_sd']:.4f}\n"
        )
        if np.isfinite(row["iterations_or_epochs_run_mean"]):
            f.write(
                f"  Iterations/epochs run: {row['iterations_or_epochs_run_mean']:.1f} ± "
                f"{row['iterations_or_epochs_run_sd']:.1f}\n"
            )
        if np.isfinite(row["cnn_best_epoch_mean"]):
            f.write(
                f"  CNN best epoch: {row['cnn_best_epoch_mean']:.1f} ± "
                f"{row['cnn_best_epoch_sd']:.1f}\n"
            )

    best_name = best_row["model"]
    op = operating_df[(operating_df.task == task_name) & (operating_df.best_model == best_name)]
    f.write(f"\nBEST MODEL BY MEAN TEST AUROC: {best_name}\n")
    f.write(
        f"  Youden threshold: {op.youden_threshold.mean():.4f} ± {op.youden_threshold.std(ddof=0):.4f}\n"
        f"  Sensitivity at specificity >=80%: "
        f"{op.sensitivity_at_specificity_ge_80.mean()*100:.2f} ± "
        f"{op.sensitivity_at_specificity_ge_80.std(ddof=0)*100:.2f}%\n"
        f"  Achieved specificity: {op.achieved_specificity_ge_80.mean()*100:.2f} ± "
        f"{op.achieved_specificity_ge_80.std(ddof=0)*100:.2f}%\n"
        f"  Threshold for specificity >=80%: "
        f"{op.threshold_for_specificity_ge_80.mean():.4f} ± "
        f"{op.threshold_for_specificity_ge_80.std(ddof=0):.4f}\n"
        f"  Specificity at sensitivity >=80%: "
        f"{op.specificity_at_sensitivity_ge_80.mean()*100:.2f} ± "
        f"{op.specificity_at_sensitivity_ge_80.std(ddof=0)*100:.2f}%\n"
        f"  Achieved sensitivity: {op.achieved_sensitivity_ge_80.mean()*100:.2f} ± "
        f"{op.achieved_sensitivity_ge_80.std(ddof=0)*100:.2f}%\n"
        f"  Threshold for sensitivity >=80%: "
        f"{op.threshold_for_sensitivity_ge_80.mean():.4f} ± "
        f"{op.threshold_for_sensitivity_ge_80.std(ddof=0):.4f}\n"
    )

print("\nSaved:")
for p in [summary_path, repeat_path, hyper_path, sig_path, operating_path, report_path, roc_data_path]:
    print(p)

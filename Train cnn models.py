# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 10:37:31 2025

@author: u5579005
"""

import torch
import torch.nn as nn
# import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix,accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt



# Best
class SmallCNN1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(229, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)
    


#%%


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

def train_cnn_with_early_stopping(
    X_train, y_train, X_val, y_val,
    max_epochs=50,
    batch_size=8
):
    model = SmallCNN1D(num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=6)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(1, max_epochs + 1):
        # -------- TRAIN --------
        model.train()
        train_losses = []
        train_preds = []
        train_true = []

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(torch.argmax(logits, 1).cpu().numpy())
            train_true.extend(yb.cpu().numpy())

        train_loss = np.mean(train_losses)
        train_acc = accuracy_score(train_true, train_preds)

        # -------- VALIDATION --------
        model.eval()
        with torch.no_grad():
            logits = model(X_val)
            val_loss = criterion(logits, y_val).item()
            val_preds = torch.argmax(logits, 1).cpu().numpy()
            val_acc = accuracy_score(y_val.cpu().numpy(), val_preds)

        # -------- STORE --------
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # -------- PRINT --------
        # print(
        #     f"Epoch {epoch:03d} | "
        #     f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
        #     f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}"
        # )

        # -------- EARLY STOPPING --------
        early_stopping.step(val_loss)
        if early_stopping.stop:
            print(f"Early stopping at epoch {epoch}")
            break

    return model, history



def plot_history(history):
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.show()

    plt.figure()
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.show()

class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x, target_class=None):
        self.model.zero_grad()

        output = self.model(x)

        if target_class is None:
            target_class = torch.argmax(output, dim=1)

        loss = output[:, target_class]
        loss.backward()

        gradients = self.gradients.detach()
        activations = self.activations.detach()

        # Global average pool over time
        weights = torch.mean(gradients, dim=2, keepdim=True)

        cam = torch.sum(weights * activations, dim=1)
        cam = torch.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy()
def plot_gradcam_1d(input_signal, cam):
    input_signal = input_signal.mean(0).cpu().numpy()  # collapse 229 channels

    plt.figure(figsize=(14,6))
    plt.plot(input_signal, label="Input Signal", alpha=0.6)

    cam_resized = np.interp(
        np.linspace(0, len(cam)-1, len(input_signal)),
        np.arange(len(cam)),
        cam
    )

    plt.plot(cam_resized * np.max(input_signal),
             label="Grad-CAM",
             linewidth=2)

    plt.legend()
    plt.title("1D Grad-CAM over Time")
    plt.show()

#%%



# data = np.load('outputs/Xy_dry-skin-no-moist.npz')
# data = np.load('outputs/Xy_dry-skin-10-moist.npz')
data = np.load('outputs/Xy_dry-skin-20-moist.npz')
# data = np.load('outputs/Xy_dry-skin-type.npz')
# data = np.load('outputs/Xy_skin-cancer.npz')


X = data['X']
y = data['y']

X = X.reshape(len(y), 400, 229, order='F')

X = torch.tensor(X, dtype=torch.float32)  # (74, 400, 229)
# X = X.unsqueeze(1)  # for 2d
X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1) #for 1d
y = torch.tensor(y, dtype=torch.long)

fpr_grid = np.linspace(0, 1, 100)
sensitivities = []
specificities = []
accuracies = []
thresholds = []

# patient IDs: two samples per patient
n_patients = len(y) // 2
unique_patients = np.arange(1, n_patients + 1)
patient_ids = np.concatenate([unique_patients, unique_patients])



test_size = 0.3
n_bootstraps = 50
rng = np.random.RandomState(42)

boot_aucs = []
bootstrap_cam0_list = []
bootstrap_cam1_list = []

for i in range(n_bootstraps):

    # ---- patient-disjoint split ----
    train_patients, test_patients = train_test_split(
        unique_patients,
        test_size=test_size,
        random_state=i
    )

    train_mask = np.isin(patient_ids, train_patients)
    test_mask  = np.isin(patient_ids, test_patients)

    # ---- bootstrap patients ----
    boot_patients = rng.choice(train_patients,size=len(train_patients), replace=True)

    bootstrap_indices = np.concatenate([
        np.where(patient_ids == p)[0] for p in boot_patients
    ])

    X_train_full = X[bootstrap_indices]
    y_train_full = y[bootstrap_indices]

    X_test = X[test_mask]
    y_test = y[test_mask]

    # ---- split TRAIN → TRAIN / VAL (patients!) ----
    train_pat, val_pat = train_test_split(
        train_patients, test_size=0.2, random_state=i
    )

    tr_mask  = np.isin(patient_ids, train_pat)
    val_mask = np.isin(patient_ids, val_pat)

    X_tr  = X[tr_mask]
    y_tr  = y[tr_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    # ---- train CNN ----
    model, history = train_cnn_with_early_stopping(
        X_tr, y_tr, X_val, y_val
    )

    # ---- test (ROC-AUC) ----
    model.eval()
    with torch.no_grad():
        y_score = torch.softmax(model(X_test), dim=1)[:, 1]

    auc = roc_auc_score(
        y_test.numpy(),
        y_score.cpu().numpy()
    )

    boot_aucs.append(auc)
    y_true = y_test
    fpr, tpr, thresh = roc_curve(y_test, y_score)
    score = roc_auc_score(y_test, y_score)
    tpr_interp = np.interp(fpr_grid, fpr, tpr)
    tpr_interp[0] = 0.0
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresh[best_idx]
    # Binarize predictions
    y_pred = (y_score >= best_thresh).int()
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Metrics
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_true, y_pred)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    accuracies.append(accuracy)
    thresholds.append(best_thresh)



    print(f"Bootstrap {i+1}/{n_bootstraps} | AUC = {auc:.4f}")
    # ---- Grad-CAM per bootstrap ----
    target_layer = model.features[6]
    gradcam = GradCAM1D(model, target_layer)
    
    # Collect indices per class (TEST SET)
    class0_indices = (y_test == 0).nonzero(as_tuple=True)[0]
    class1_indices = (y_test == 1).nonzero(as_tuple=True)[0]
    
    n_samples = min(10, len(class0_indices), len(class1_indices))
    
    class0_samples = X_test[class0_indices[:n_samples]]
    class1_samples = X_test[class1_indices[:n_samples]]
    
    def get_mean_cam(samples, target_class):
        cams = []
        for x in samples:
            x = x.unsqueeze(0)
            cam = gradcam.generate(x, target_class=target_class)
    
            # resize to original time length (400)
            input_signal = x.squeeze(0).mean(0).cpu().numpy()
            cam_resized = np.interp(
                np.linspace(0, len(cam)-1, len(input_signal)),
                np.arange(len(cam)),
                cam
            )
    
            cams.append(cam_resized)
    
        return np.mean(cams, axis=0)
    
    mean_cam0 = get_mean_cam(class0_samples, 0)
    mean_cam1 = get_mean_cam(class1_samples, 1)
    
    bootstrap_cam0_list.append(mean_cam0)
    bootstrap_cam1_list.append(mean_cam1)

boot_aucs = np.array(boot_aucs)
print(f"CNN, {boot_aucs.mean()*100:.2f}, {boot_aucs.std()*100:.2f}")

# np.savez('outputs/AUROC_dry-skin-no-moist', auroc = boot_aucs)
# np.savez('outputs/AUROC_dry-skin-10-moist', auroc = boot_aucs)
np.savez('outputs/AUROC_dry-skin-20-moist', auroc = boot_aucs)
# np.savez('outputs/AUROC_dry-skin-type', auroc = boot_aucs)
# np.savez('outputs/AUROC_skin-cancer', auroc = boot_aucs)

print(model)

bootstrap_cam0_array = np.stack(bootstrap_cam0_list)
bootstrap_cam1_array = np.stack(bootstrap_cam1_list)

mean_cam0 = bootstrap_cam0_array.mean(axis=0)
std_cam0  = bootstrap_cam0_array.std(axis=0)

mean_cam1 = bootstrap_cam1_array.mean(axis=0)
std_cam1  = bootstrap_cam1_array.std(axis=0)

# Convert to arrays
sensitivities = np.array(sensitivities)
specificities = np.array(specificities)
accuracies = np.array(accuracies)
thresholds = np.array(thresholds)

# np.savez('outputs/AUROC_dry-skin-type', auroc = bootstrap_auroc)
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

#%% plot gradcam
plt.rcdefaults()
# === Set Global Plot Style ===
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'legend.fontsize': 14,
    'lines.linewidth': 3,
    'axes.linewidth': 1.5,
    'savefig.dpi': 300
})

plt.figure(figsize=(6, 8))

time = np.arange(len(mean_cam0))

# Compute mean input signal across same samples (collapse channels)
mean_signal = X.mean(0).mean(0).cpu().numpy()  # shape: (time,)

# Optionally normalize signal to match CAM scale
signal_scaled = mean_signal / mean_signal.max() * max(mean_cam0.max(), mean_cam1.max())

# Plot mean input signal
plt.plot(time, signal_scaled, label="Mean Input Signal", color='gray', linewidth=2, linestyle='--')

# Class 0 Grad-CAM
# plt.plot(time, mean_cam0, label="Healthy", color='blue', linewidth=2)
plt.plot(time, mean_cam0, label="Eczema", color='blue', linewidth=2)
plt.fill_between(time,
                 mean_cam0 - std_cam0,
                 mean_cam0 + std_cam0,
                 alpha=0.3)

# Class 1 Grad-CAM
# plt.plot(time, mean_cam1, label="Dry", color='red', linewidth=2)
plt.plot(time, mean_cam1, label="Psoriasis", color='red', linewidth=2)
# plt.plot(time, mean_cam1, label="Skin cancer", color='red', linewidth=2)
plt.fill_between(time,
                 mean_cam1 - std_cam1,
                 mean_cam1 + std_cam1,
                 alpha=0.3)

# plt.title("Grad-CAM Mean ± STD Across Bootstraps with Mean Input Signal")
plt.xlabel("Data points")
plt.ylabel("Normalized Activation")
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


#%% p value for comparison of model performance for different moisturisation
data = np.load('outputs/AUROC_dry-skin-no-moist.npz')
bootstrap_auroc_0 = data['auroc']
data = np.load('outputs/AUROC_dry-skin-10-moist.npz')
bootstrap_auroc_1 = data['auroc']
data = np.load('outputs/AUROC_dry-skin-20-moist.npz')
bootstrap_auroc_2 = data['auroc']


from scipy.stats import  wilcoxon
print("Model1 vs Model2:", wilcoxon(bootstrap_auroc_0, bootstrap_auroc_1))
print("Model1 vs Model2:", wilcoxon(bootstrap_auroc_0, bootstrap_auroc_2))
print("Model1 vs Model2:", wilcoxon(bootstrap_auroc_2, bootstrap_auroc_1))


#%% Ablation experiments
    
# # Best with dropout
# class SmallCNN1D(nn.Module):
#     def __init__(self, num_classes=2):
#         super().__init__()

#         self.features = nn.Sequential(
#                     nn.Conv1d(229, 64, 3, padding=1),
#                     nn.ReLU(),
#                     nn.MaxPool1d(2),

#                     nn.Conv1d(64, 128, 3, padding=1),
#                     nn.ReLU(),
#                     nn.MaxPool1d(2),

#                     nn.Conv1d(128, 256, 3, padding=1),
#                     nn.ReLU(),
#                     nn.MaxPool1d(2),
#                 )


#         self.gap = nn.AdaptiveAvgPool1d(1)
#         self.dropout = nn.Dropout(0.5)
#         self.fc = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.gap(x).squeeze(-1)
#         x = self.dropout(x)
#         return self.fc(x)


# class SmallCNN1D(nn.Module):
#     def __init__(self, num_classes=2):
#         super().__init__()

#         self.features = nn.Sequential(
#                     nn.Conv1d(229, 32, 3, padding=1),
#                     nn.ReLU(),
#                     nn.MaxPool1d(2),

#                     nn.Conv1d(32, 128, 3, padding=1),
#                     nn.ReLU(),
#                     nn.MaxPool1d(2),

#                     # nn.Conv1d(128, 256, 3, padding=1),
#                     # nn.ReLU(),
#                     # nn.MaxPool1d(2),
#                 )


#         self.gap = nn.AdaptiveAvgPool1d(1)
#         self.dropout = nn.Dropout(0.5)
#         self.fc = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.gap(x).squeeze(-1)
#         x = self.dropout(x)
#         return self.fc(x)


# class SmallCNN1D(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         self.features = nn.Sequential(
#             nn.Conv1d(229, 64, 9, padding=3),
#             nn.ReLU(),
#             nn.MaxPool1d(2),

#             nn.Conv1d(64, 128, 9, padding=3),
#             nn.ReLU(),
#             nn.MaxPool1d(2),

#             nn.Conv1d(128, 256, 9, padding=3),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#         )

#         self.gap = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.gap(x).squeeze(-1)
#         return self.fc(x)
    

# class SmallCNN1D_975432(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         self.features = nn.Sequential(
#             nn.Conv1d(229, 64, 9, padding=4),
#             nn.ReLU(),
#             nn.MaxPool1d(2),

#             nn.Conv1d(64, 128, 7, padding=3),
#             nn.ReLU(),
#             nn.MaxPool1d(2),

#             nn.Conv1d(128, 256, 5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#         )

#         self.gap = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.gap(x).squeeze(-1)
#         return self.fc(x)

# class SmallCNN1D(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         self.features = nn.Sequential(
#             nn.Conv1d(229, 64, 40, padding=3),
#             nn.ReLU(),
#             nn.MaxPool1d(2),

#             nn.Conv1d(64, 128, 40, padding=3),
#             nn.ReLU(),
#             nn.MaxPool1d(2),

#             nn.Conv1d(128, 256, 40, padding=3),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#         )

#         self.gap = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.gap(x).squeeze(-1)
#         return self.fc(x)

    
# class DilatedMixingCNN(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         self.features = nn.Sequential(
#             nn.Conv1d(229, 64, kernel_size=7, padding=3),
#             nn.ReLU(),

#             nn.Conv1d(64, 128, kernel_size=5, padding=4, dilation=2),
#             nn.ReLU(),

#             nn.Conv1d(128, 128, kernel_size=5, padding=8, dilation=4),
#             nn.ReLU(),

#             nn.AdaptiveAvgPool1d(1)
#         )

#         self.fc = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.features(x).squeeze(-1)
#         return self.fc(x)



# class SpectralTemporalCNN(nn.Module):
#     def __init__(self, num_classes=2):
#         super().__init__()

#         # -------- SPECTRAL ENCODER --------
#         # Applied to each spectrum (400 bins)
#         self.spectral = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=7, padding=3),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(2),

#             nn.Conv1d(32, 64, kernel_size=5, padding=2),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),

#             nn.AdaptiveAvgPool1d(1)   # → (B*T, 64, 1)
#         )

#         # -------- TEMPORAL CNN --------
#         # Operates across 229 time points
#         self.temporal = nn.Sequential(
#             nn.Conv1d(64, 64, kernel_size=5, padding=2),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),

#             nn.Conv1d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(),

#             nn.AdaptiveAvgPool1d(1)   # → (B, 64, 1)
#         )

#         self.dropout = nn.Dropout(0.5)
#         self.fc = nn.Linear(64, num_classes)

#     def forward(self, x):
#         # x shape: (B, 229, 400)
#         B, T, S = x.shape

#         # ----- Spectral stage -----
#         x = x.view(B*T, 1, S)              # (B*T, 1, 400)
#         feat = self.spectral(x)            # (B*T, 64, 1)
#         feat = feat.squeeze(-1)            # (B*T, 64)

#         # reshape back to time structure
#         feat = feat.view(B, T, 64)         # (B, 229, 64)

#         # ----- Temporal stage -----
#         feat = feat.permute(0, 2, 1)       # (B, 64, 229)
#         feat = self.temporal(feat)         # (B, 64, 1)

#         feat = feat.squeeze(-1)            # (B, 64)
#         feat = self.dropout(feat)

#         return self.fc(feat)

# class SpectralTemporalStats(nn.Module):
#     def __init__(self, num_classes=2):
#         super().__init__()

#         self.spectral = nn.Sequential(
#             nn.Conv1d(1, 32, 7, padding=3),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(2),

#             nn.Conv1d(32, 64, 5, padding=2),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),

#             nn.AdaptiveAvgPool1d(1)
#         )

#         self.dropout = nn.Dropout(0.5)
#         self.fc = nn.Linear(64*2, num_classes)

#     def forward(self, x):
#         B, T, S = x.shape

#         x = x.view(B*T, 1, S)
#         feat = self.spectral(x).squeeze(-1)   # (B*T, 64)
#         feat = feat.view(B, T, 64)            # (B, 229, 64)

#         mean_feat = feat.mean(dim=1)
#         std_feat  = feat.std(dim=1)

#         combined = torch.cat([mean_feat, std_feat], dim=1)

#         combined = self.dropout(combined)
#         return self.fc(combined)

# class MultiScaleMixCNN(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         self.branch1 = nn.Conv1d(229, 64, 3, padding=1)
#         self.branch2 = nn.Conv1d(229, 64, 7, padding=3)
#         self.branch3 = nn.Conv1d(229, 64, 15, padding=7)

#         self.combine = nn.Conv1d(192, 128, 1)

#         self.pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(128, num_classes)

#     def forward(self, x):
#         b1 = torch.relu(self.branch1(x))
#         b2 = torch.relu(self.branch2(x))
#         b3 = torch.relu(self.branch3(x))

#         x = torch.cat([b1, b2, b3], dim=1)
#         x = torch.relu(self.combine(x))
#         x = self.pool(x).squeeze(-1)
#         return self.fc(x)




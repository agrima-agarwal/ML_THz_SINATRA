# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


N_REPEATS = 50
OUTER_TEST_SIZE = 0.30
INNER_VALIDATION_SIZE = 0.20
SEED_BASE = 1000

CNN_LR = 1e-3
CNN_WEIGHT_DECAY = 1e-4
CNN_KERNEL_SIZE = 3
CNN_DROPOUT = 0.0
CNN_BATCH_SIZE = 8
CNN_MAX_EPOCHS = 50
CNN_PATIENCE = 6
CNN_MIN_DELTA = 1e-4

N_CAM_SAMPLES_PER_CLASS = 10

N_INDIVIDUAL_EXAMPLES = 4
INDIVIDUAL_REPEAT_TO_PLOT = 0

MODEL_DIR = Path("outputs/gradcam/gradcam_models") 
MODEL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path("outputs/gradcam/gradcam_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FORCE_RETRAIN = False

TASKS = [
    ("Dry vs Healthy - no moisturizer",
     "outputs/Xy/Xy_dry-skin-no-moist.npz",
     "paired_regions",
     "Healthy",
     "Dry"),

    ("Dry vs Healthy - 10 min moisturizer",
     "outputs/Xy/Xy_dry-skin-10-moist.npz",
     "paired_regions",
     "Healthy",
     "Dry"),

    ("Dry vs Healthy - 20 min moisturizer",
     "outputs/Xy/Xy_dry-skin-20-moist.npz",
     "paired_regions",
     "Healthy",
     "Dry"),

    ("Eczema vs Psoriasis",
     "outputs/Xy/Xy_dry-skin-type.npz",
     "single_label",
     "Eczema",
     "Psoriasis"),

    ("Skin Cancer vs Healthy",
     "outputs/Xy/Xy_skin-cancer.npz",
     "paired_regions",
     "Healthy",
     "Skin cancer"),
]

TASKS_TO_RUN = [t[0] for t in TASKS]


# Set NumPy and PyTorch random seeds.
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        raise ValueError("Unknown structure")

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


# Two-block 1D CNN used for the final classifier.
class TwoBlockCNN1D(nn.Module):
    # Build the two-block CNN.
    def __init__(self, input_channels=229, input_length=400):
        super().__init__()

        p = CNN_KERNEL_SIZE // 2

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, CNN_KERNEL_SIZE, padding=p),  # 0
            nn.ReLU(),                                                  # 1
            nn.MaxPool1d(2),                                            # 2
            nn.Conv1d(64, 128, CNN_KERNEL_SIZE, padding=p),             # 3 <- Grad-CAM
            nn.ReLU(),                                                  # 4
            nn.MaxPool1d(2),                                            # 5
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
        model.parameters(),
        lr=CNN_LR,
        weight_decay=CNN_WEIGHT_DECAY,
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
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0.0
        n = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
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


# Save a CNN checkpoint and split metadata.
def save_model_checkpoint(
    model,
    model_path,
    repeat_i,
    seed,
    train_patients,
    val_patients,
    test_patients,
    epochs_run,
    best_epoch,
    best_val_loss,
):
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "repeat_i": int(repeat_i),
            "seed": int(seed),
            "train_patients": np.asarray(train_patients),
            "val_patients": np.asarray(val_patients),
            "test_patients": np.asarray(test_patients),
            "epochs_run": int(epochs_run),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "input_channels": 229,
            "input_length": 400,
        },
        model_path,
    )


# Load a saved CNN checkpoint.
def load_model_checkpoint(model_path, device):
    try:
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            model_path,
            map_location=device,
        )

    model = TwoBlockCNN1D(
        input_channels=int(checkpoint.get("input_channels", 229)),
        input_length=int(checkpoint.get("input_length", 400)),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


# TEST METRICS

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
    y_pred = (np.asarray(scores) >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    accuracy = accuracy_score(y_true, y_pred)

    return float(sensitivity), float(specificity), float(accuracy)


# Compute one-dimensional Grad-CAM maps.
class GradCAM1D:

    # Register hooks on the selected convolutional layer.
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )
        self.backward_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    # Store activations from the target layer.
    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    # Store gradients from the target layer.
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    # Generate a one-dimensional Grad-CAM map.
    def generate(self, x, mode="contrast", target_class=None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)

        if mode == "contrast":
            score = logits[0, 1] - logits[0, 0]

        elif mode == "class":
            if target_class is None:
                target_class = int(torch.argmax(logits, dim=1).item())
            score = logits[0, int(target_class)]

        else:
            raise ValueError("mode must be 'contrast' or 'class'")

        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=2, keepdim=True)
        cam = (weights * activations).sum(dim=1).squeeze(0)

        if mode == "class":
            cam = torch.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

        else:
            max_abs = cam.abs().max()
            cam = cam / (max_abs + 1e-8)

        return cam.cpu().numpy()

    # Remove the registered Grad-CAM hooks.
    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


# Interpolate a Grad-CAM map to the original signal length.
def resize_cam(cam, input_length=400):
    return np.interp(
        np.linspace(0, len(cam) - 1, input_length),
        np.arange(len(cam)),
        cam,
    )


# Average class-specific Grad-CAM maps across selected samples.
def class_specific_mean_cam(
    gradcam,
    X_test,
    y_test,
    target_class,
    device,
    n_samples=None,
):
    class_indices = np.where(y_test == target_class)[0]

    if n_samples is not None:
        class_indices = class_indices[:n_samples]

    cams = []

    for idx in class_indices:
        x = X_test[idx:idx + 1].to(device)

        cam = gradcam.generate(
            x,
            mode="class",
            target_class=target_class,
        )

        cams.append(
            resize_cam(cam, X_test.shape[-1])
        )

    return np.mean(np.stack(cams), axis=0)


# Plot class-specific Grad-CAM profiles with the mean THz signal.
def plot_class_cams(task_name, class0_name, class1_name,
                    mean_class0_cam, std_class0_cam,
                    mean_class1_cam, std_class1_cam,
                    mean_signal, std_signal, output_path):
    plt.rcdefaults()
    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 16, "axes.titlesize": 17,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "xtick.direction": "in", "ytick.direction": "in",
        "axes.linewidth": 1.4, "savefig.dpi": 300,
    })

    x = np.arange(len(mean_class0_cam))
    signal_scale = 0.85 / (np.max(np.abs(mean_signal)) + 1e-8)
    signal_scaled = mean_signal * signal_scale
    std_signal_scaled = std_signal * signal_scale

    fig, ax = plt.subplots(figsize=(6, 8))

    ax.fill_between(
        x,
        signal_scaled - std_signal_scaled,
        signal_scaled + std_signal_scaled,
        color="grey",
        alpha=0.5,
        linewidth=0,
    )
    ax.plot(
        x,
        signal_scaled,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="Mean THz signal",
    )

    ax.fill_between(
        x,
        mean_class0_cam - std_class0_cam,
        mean_class0_cam + std_class0_cam,
        color="blue",
        alpha=0.5,
        linewidth=0,
    )
    ax.plot(
        x,
        mean_class0_cam,
        color="blue",
        linewidth=2.5,
        label=class0_name,
    )

    ax.fill_between(
        x,
        mean_class1_cam - std_class1_cam,
        mean_class1_cam + std_class1_cam,
        color="red",
        alpha=0.5,
        linewidth=0,
    )
    ax.plot(
        x,
        mean_class1_cam,
        color="red",
        linewidth=2.5,
        label=class1_name,
    )

    ax.set_xlabel("Data points")
    ax.set_ylabel("Grad-CAM importance")
    ax.set_xlim([0, 400])
    ax.legend(loc="lower left", framealpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# Run the full analysis for one classification task.
def run_task(task_name, data_file, structure, class0_name, class1_name):
    print("\n" + "=" * 90)
    print(task_name)
    print("=" * 90)

    data = np.load(data_file)
    X = data["X"]
    y = data["y"].astype(int)
    

    unique_patients, patient_ids = make_patient_ids(y, structure)
    X_cnn = prepare_cnn_X(X, y)

    sample_signals = X_cnn.mean(dim=1).cpu().numpy()   # shape: (N_samples, 400)

    # Mean and true sample-to-sample standard deviation of the THz trace.
    mean_signal = sample_signals.mean(axis=0)
    std_signal = sample_signals.std(axis=0)

    repeat_aucs = []
    sensitivities = []
    specificities = []
    accuracies = []
    thresholds = []

    repeat_class0_cam = []
    repeat_class1_cam = []

    for repeat_i in range(N_REPEATS):
        train_patients, val_patients, test_patients = split_patients(
            y,
            unique_patients,
            structure,
            repeat_i,
        )

        train_mask = np.isin(patient_ids, train_patients)
        val_mask = np.isin(patient_ids, val_patients)
        test_mask = np.isin(patient_ids, test_patients)

        X_train = X_cnn[train_mask]
        y_train = y[train_mask]

        X_val = X_cnn[val_mask]
        y_val = y[val_mask]

        X_test = X_cnn[test_mask]
        y_test = y[test_mask]

        stem = (
            task_name.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
        )

        model_path = (
            MODEL_DIR
            / stem
            / f"repeat_{repeat_i:02d}.pt"
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if model_path.exists() and not FORCE_RETRAIN:
            model, checkpoint = load_model_checkpoint(
                model_path,
                device,
            )

            epochs_run = int(checkpoint.get("epochs_run", -1))
            best_epoch = int(checkpoint.get("best_epoch", -1))
            best_val_loss = float(
                checkpoint.get("best_val_loss", np.nan)
            )

            saved_train = np.asarray(
                checkpoint.get("train_patients", train_patients)
            )
            saved_val = np.asarray(
                checkpoint.get("val_patients", val_patients)
            )
            saved_test = np.asarray(
                checkpoint.get("test_patients", test_patients)
            )

            if not np.array_equal(saved_train, np.asarray(train_patients)):
                raise RuntimeError(
                    f"Saved train split does not match repeat {repeat_i}"
                )

            if not np.array_equal(saved_val, np.asarray(val_patients)):
                raise RuntimeError(
                    f"Saved validation split does not match repeat {repeat_i}"
                )

            if not np.array_equal(saved_test, np.asarray(test_patients)):
                raise RuntimeError(
                    f"Saved test split does not match repeat {repeat_i}"
                )

            model_source = "loaded"

        else:
            model, device, epochs_run, best_epoch, best_val_loss = train_cnn(
                X_train,
                y_train,
                X_val,
                y_val,
                SEED_BASE + repeat_i,
            )

            save_model_checkpoint(
                model=model,
                model_path=model_path,
                repeat_i=repeat_i,
                seed=SEED_BASE + repeat_i,
                train_patients=train_patients,
                val_patients=val_patients,
                test_patients=test_patients,
                epochs_run=epochs_run,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
            )

            model_source = "trained + saved"

        # Test metrics, unchanged.
        test_scores = cnn_score(model, device, X_test)
        test_auc = roc_auc_score(y_test, test_scores)

        threshold = youden_threshold(y_test, test_scores)
        sens, spec, acc = metrics_at_threshold(
            y_test,
            test_scores,
            threshold,
        )

        repeat_aucs.append(test_auc)
        thresholds.append(threshold)
        sensitivities.append(sens)
        specificities.append(spec)
        accuracies.append(acc)

        gradcam = GradCAM1D(
            model,
            model.features[3],
        )

        # class-specific CAMs, for direct comparison.
        cam0 = class_specific_mean_cam(
            gradcam,
            X_test,
            y_test,
            target_class=0,
            device=device,
            n_samples=N_CAM_SAMPLES_PER_CLASS,
        )

        cam1 = class_specific_mean_cam(
            gradcam,
            X_test,
            y_test,
            target_class=1,
            device=device,
            n_samples=N_CAM_SAMPLES_PER_CLASS,
        )

        repeat_class0_cam.append(cam0)
        repeat_class1_cam.append(cam1)

        gradcam.remove_hooks()

        print(
            f"Repeat {repeat_i + 1:02d}/{N_REPEATS} | "
            f"AUC={test_auc:.4f} | "
            f"best epoch={best_epoch} | "
            f"{model_source}"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    repeat_aucs = np.asarray(repeat_aucs)
    sensitivities = np.asarray(sensitivities)
    specificities = np.asarray(specificities)
    accuracies = np.asarray(accuracies)
    thresholds = np.asarray(thresholds)

    repeat_class0_cam = np.stack(repeat_class0_cam)
    repeat_class1_cam = np.stack(repeat_class1_cam)

    mean_class0_cam = repeat_class0_cam.mean(axis=0)
    std_class0_cam = repeat_class0_cam.std(axis=0)

    mean_class1_cam = repeat_class1_cam.mean(axis=0)
    std_class1_cam = repeat_class1_cam.std(axis=0)

    print(
        f"\nCNN AUROC: {repeat_aucs.mean():.3f} ± {repeat_aucs.std():.3f}"
    )

    stem = (
        task_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )

    # Save all CAM outputs.
    np.savez(
        OUTPUT_DIR / f"{stem}_gradcam_classes.npz",
        repeat_auc=repeat_aucs,
        sensitivity=sensitivities,
        specificity=specificities,
        accuracy=accuracies,
        threshold=thresholds,

        repeat_class0_cam=repeat_class0_cam,
        repeat_class1_cam=repeat_class1_cam,

        mean_class0_cam=mean_class0_cam,
        std_class0_cam=std_class0_cam,

        mean_class1_cam=mean_class1_cam,
        std_class1_cam=std_class1_cam,

        mean_input_signal=mean_signal,
        std_input_signal=std_signal,
    )

    plot_class_cams(
        task_name,
        class0_name,
        class1_name,
        mean_class0_cam,
        std_class0_cam,
        mean_class1_cam,
        std_class1_cam,
        mean_signal,
        std_signal,
        OUTPUT_DIR / f"{stem}_class_gradcam.png",
    )


# Run the script.
def main():
    print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")
    print("Running final CNN class-specific Grad-CAM analysis.")
    print("Same 56/14/30 patient-independent split logic as final analysis.")
    print("No bootstrap resampling.")
    print("Saved CNN checkpoints are reused automatically when available.")

    for task in TASKS:
        if task[0] in TASKS_TO_RUN:
            run_task(*task)


if __name__ == "__main__":
    main()

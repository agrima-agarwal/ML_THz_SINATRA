# -*- coding: utf-8 -*-

import copy
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    confusion_matrix,
)

from scipy.stats import wilcoxon


# Set random seeds for reproducible training.
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


test_size = 0.30
validation_size = 0.20
n_repeats = 50

batch_size = 8
max_epochs = 50
early_stopping_patience = 6

learning_rate = 1e-3
weight_decay = 1e-4
kernel_size = 3
dropout = 0.0

OUTPUT_DIR = Path("outputs/cnn_ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


TASKS = [
    {
        "name": "dry_no_moist",
        "label": "Dry vs Healthy - no moisturizer",
        "file": "outputs/Xy/Xy_dry-skin-no-moist.npz",
        "structure": "paired_regions",
    },
    {
        "name": "dry_10_moist",
        "label": "Dry vs Healthy - 10 min moisturizer",
        "file": "outputs/Xy/Xy_dry-skin-10-moist.npz",
        "structure": "paired_regions",
    },
    {
        "name": "dry_20_moist",
        "label": "Dry vs Healthy - 20 min moisturizer",
        "file": "outputs/Xy/Xy_dry-skin-20-moist.npz",
        "structure": "paired_regions",
    },
    {
        "name": "eczema_psoriasis",
        "label": "Eczema vs Psoriasis",
        "file": "outputs/Xy/Xy_dry-skin-type.npz",
        "structure": "single_label",
    },
    {
        "name": "skin_cancer",
        "label": "Skin Cancer vs Healthy",
        "file": "outputs/Xy/Xy_skin-cancer.npz",
        "structure": "paired_regions",
    },
]


ablation_variants = [
    "3_layer",
    "4_layer",
    "2_layer",
    "no_max_pooling",
    "global_avg_p",
]

REFERENCE_VARIANT = "3_layer"


# Configurable 1D CNN used for the architecture ablation study.
class AblationCNN1D(nn.Module):

    # Build the selected CNN ablation architecture.
    def __init__(
        self,
        variant,
        num_classes=2,
        kernel_size=3,
        dropout=0.0,
        input_length=400,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("Use an odd kernel size.")

        self.variant = variant
        padding = kernel_size // 2

        if variant == "4_layer":
            channels = [229, 64, 128, 256]
            use_pooling = True
            use_gap = False

        elif variant == "3_layer":
            channels = [229, 64, 128]
            use_pooling = True
            use_gap = False

        elif variant == "2_layer":
            channels = [229, 64]
            use_pooling = True
            use_gap = False

        elif variant == "no_max_pooling":
            channels = [229, 64, 128]
            use_pooling = False
            use_gap = False

        elif variant == "global_avg_p":
            channels = [229, 64, 128]
            use_pooling = True
            use_gap = True

        else:
            raise ValueError(f"Unknown ablation variant: {variant}")

        layers = []

        for block_i in range(len(channels) - 1):
            layers.append(
                nn.Conv1d(
                    channels[block_i],
                    channels[block_i + 1],
                    kernel_size,
                    padding=padding,
                )
            )
            layers.append(nn.ReLU())

            if use_pooling:
                layers.append(nn.MaxPool1d(2))

        self.features = nn.Sequential(*layers)
        final_channels = channels[-1]

        self.dropout = (
            nn.Dropout(dropout)
            if dropout > 0
            else nn.Identity()
        )

        self.use_gap = use_gap

        if use_gap:
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(final_channels, num_classes)

        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 229, input_length)
                feature_shape = self.features(dummy).shape

            flattened_features = feature_shape[1] * feature_shape[2]

            self.pool = nn.Identity()
            self.fc = nn.Linear(flattened_features, num_classes)

    # Run the forward pass.
    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)

        if self.use_gap:
            x = self.pool(x).squeeze(-1)
        else:
            x = torch.flatten(x, start_dim=1)

        return self.fc(x)


# Track validation loss for early stopping.
class EarlyStopping:
    # Initialize early-stopping parameters and state.
    def __init__(self, patience=6, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.stop = False

    # Update the early-stopping state.
    def step(self, val_loss):
        improved = val_loss < self.best_loss - self.min_delta

        if improved:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

        return improved


# Train one CNN ablation variant with early stopping.
def train_ablation_model(
    variant,
    X_train,
    y_train,
    X_validation,
    y_validation,
    seed,
):
    set_all_seeds(seed)

    model = AblationCNN1D(
        variant=variant,
        num_classes=2,
        kernel_size=kernel_size,
        dropout=dropout,
        input_length=X_train.shape[-1],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=1e-4,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )

    X_validation_device = X_validation.to(device)
    y_validation_device = y_validation.to(device)

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):

        model.train()

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            validation_logits = model(X_validation_device)
            validation_loss = criterion(
                validation_logits,
                y_validation_device,
            ).item()

        improved = early_stopping.step(validation_loss)

        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        if early_stopping.stop:
            break

    model.load_state_dict(best_state)

    return model, best_epoch


# Return class-1 softmax scores from the CNN.
def predict_scores(model, X):
    model.eval()

    with torch.no_grad():
        logits = model(X.to(device))
        scores = torch.softmax(logits, dim=1)[:, 1]

    return scores.cpu().numpy()


# Select the threshold that maximises Youden's J statistic.
def youden_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    finite = np.isfinite(thresholds)
    fpr = fpr[finite]
    tpr = tpr[finite]
    thresholds = thresholds[finite]

    if len(thresholds) == 0:
        return 0.5

    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


# Calculate accuracy, sensitivity and specificity.
def binary_metrics(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    accuracy = accuracy_score(y_true, pred)

    return accuracy, sensitivity, specificity


# Apply Holm correction to multiple p-values.
def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)

    if m == 0:
        return p_values

    order = np.argsort(p_values)
    adjusted_sorted = np.empty(m, dtype=float)

    running_max = 0.0

    for rank, idx in enumerate(order):
        corrected = (m - rank) * p_values[idx]
        corrected = min(corrected, 1.0)

        running_max = max(running_max, corrected)
        adjusted_sorted[rank] = running_max

    adjusted = np.empty(m, dtype=float)

    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]

    return adjusted


# LOAD TASK DATA

# Load one task and reshape it for CNN input.
def load_task(task):
    data = np.load(task["file"])

    if "X" in data.files:
        X_flat = data["X"]
    elif "x" in data.files:
        X_flat = data["x"]
    else:
        raise KeyError(f"{task['file']}: could not find X/x.")

    if "y" in data.files:
        y_np = data["y"].astype(int)
    elif "Y" in data.files:
        y_np = data["Y"].astype(int)
    else:
        raise KeyError(f"{task['file']}: could not find y/Y.")

    structure = task["structure"]

    if structure == "single_label":
        n_patients = len(y_np)
        unique_patients = np.arange(n_patients)
        patient_ids = unique_patients.copy()

    elif structure == "paired_regions":
        if len(y_np) % 2 != 0:
            raise ValueError(
                f"{task['file']}: paired_regions expects an even number of samples."
            )

        n_patients = len(y_np) // 2
        unique_patients = np.arange(n_patients)

        patient_ids = np.concatenate(
            [unique_patients, unique_patients]
        )

    else:
        raise ValueError(
            "structure must be 'paired_regions' or 'single_label'."
        )

    X_cnn_np = X_flat.reshape(len(y_np), 400, 229, order="F")
    X_cnn_np = np.transpose(X_cnn_np, (0, 2, 1))

    X_cnn = torch.tensor(X_cnn_np, dtype=torch.float32)
    y_cnn = torch.tensor(y_np, dtype=torch.long)

    return (
        X_cnn,
        y_cnn,
        y_np,
        patient_ids,
        unique_patients,
        n_patients,
    )


# Run the full analysis for one classification task.
def run_task(task):

    print("\n" + "#" * 100)
    print(task["label"])
    print("#" * 100)

    (
        X_cnn,
        y_cnn,
        y_np,
        patient_ids,
        unique_patients,
        n_patients,
    ) = load_task(task)

    print("File:", task["file"])
    print("Structure:", task["structure"])
    print("Samples:", len(y_np))
    print("Patients:", n_patients)
    print("Class counts:", np.unique(y_np, return_counts=True))
    print("CNN shape:", tuple(X_cnn.shape))

    results = {
        variant: {
            "validation_auc": [],
            "test_auc": [],
            "accuracy": [],
            "sensitivity": [],
            "specificity": [],
            "threshold": [],
            "best_epoch": [],
            "n_parameters": [],
        }
        for variant in ablation_variants
    }

    for repeat_i in range(n_repeats):

        print(
            f"\n{task['name']} | "
            f"repetition {repeat_i + 1}/{n_repeats}"
        )

        if task["structure"] == "single_label":

            train_pool_patients, test_patients = train_test_split(
                unique_patients,
                test_size=test_size,
                random_state=repeat_i,
                stratify=y_np,
            )

            train_patients, validation_patients = train_test_split(
                train_pool_patients,
                test_size=validation_size,
                random_state=repeat_i,
                stratify=y_np[train_pool_patients],
            )

        else:

            train_pool_patients, test_patients = train_test_split(
                unique_patients,
                test_size=test_size,
                random_state=repeat_i,
            )

            train_patients, validation_patients = train_test_split(
                train_pool_patients,
                test_size=validation_size,
                random_state=repeat_i,
            )

        train_mask = np.isin(patient_ids, train_patients)
        validation_mask = np.isin(patient_ids, validation_patients)
        test_mask = np.isin(patient_ids, test_patients)

        X_train = X_cnn[train_mask]
        y_train = y_cnn[train_mask]

        X_validation = X_cnn[validation_mask]
        y_validation = y_cnn[validation_mask]

        X_test = X_cnn[test_mask]
        y_test = y_cnn[test_mask]

        y_validation_np = y_validation.numpy()
        y_test_np = y_test.numpy()

        # Same initialization/shuffle seed for all variants within a split.
        training_seed = 1000 + repeat_i

        for variant in ablation_variants:

            model, best_epoch = train_ablation_model(
                variant=variant,
                X_train=X_train,
                y_train=y_train,
                X_validation=X_validation,
                y_validation=y_validation,
                seed=training_seed,
            )

            validation_scores = predict_scores(
                model,
                X_validation,
            )

            test_scores = predict_scores(
                model,
                X_test,
            )

            validation_auc = roc_auc_score(
                y_validation_np,
                validation_scores,
            )

            test_auc = roc_auc_score(
                y_test_np,
                test_scores,
            )

            threshold = youden_threshold(
                y_validation_np,
                validation_scores,
            )

            accuracy, sensitivity, specificity = binary_metrics(
                y_test_np,
                test_scores,
                threshold,
            )

            n_parameters = sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )

            results[variant]["validation_auc"].append(validation_auc)
            results[variant]["test_auc"].append(test_auc)
            results[variant]["accuracy"].append(accuracy)
            results[variant]["sensitivity"].append(sensitivity)
            results[variant]["specificity"].append(specificity)
            results[variant]["threshold"].append(threshold)
            results[variant]["best_epoch"].append(best_epoch)
            results[variant]["n_parameters"].append(n_parameters)

            print(
                f"  {variant:16s} | "
                f"val={validation_auc:.4f} | "
                f"test={test_auc:.4f} | "
                f"epoch={best_epoch:2d}"
            )


    summary_rows = []

    for variant in ablation_variants:

        r = results[variant]

        row = {
            "task": task["name"],
            "task_label": task["label"],
            "variant": variant,
            "parameters": int(np.median(r["n_parameters"])),
            "validation_auc_mean": np.mean(r["validation_auc"]),
            "validation_auc_sd": np.std(r["validation_auc"]),
            "test_auc_mean": np.mean(r["test_auc"]),
            "test_auc_sd": np.std(r["test_auc"]),
            "accuracy_mean": np.nanmean(r["accuracy"]),
            "accuracy_sd": np.nanstd(r["accuracy"]),
            "sensitivity_mean": np.nanmean(r["sensitivity"]),
            "sensitivity_sd": np.nanstd(r["sensitivity"]),
            "specificity_mean": np.nanmean(r["specificity"]),
            "specificity_sd": np.nanstd(r["specificity"]),
            "best_epoch_mean": np.mean(r["best_epoch"]),
            "best_epoch_sd": np.std(r["best_epoch"]),
        }

        summary_rows.append(row)


    reference = np.asarray(
        results[REFERENCE_VARIANT]["test_auc"]
    )

    wilcoxon_rows = []

    for variant in ablation_variants:

        if variant == REFERENCE_VARIANT:
            continue

        comparison = np.asarray(
            results[variant]["test_auc"]
        )

        difference = comparison - reference

        try:
            statistic, p_value = wilcoxon(
                comparison,
                reference,
                alternative="two-sided",
                zero_method="wilcox",
            )
        except ValueError:
            statistic = 0.0
            p_value = 1.0

        wilcoxon_rows.append(
            {
                "task": task["name"],
                "task_label": task["label"],
                "reference": REFERENCE_VARIANT,
                "variant": variant,
                "mean_delta_auc": np.mean(difference),
                "median_delta_auc": np.median(difference),
                "wilcoxon_statistic": statistic,
                "p_raw": p_value,
            }
        )

    adjusted = holm_adjust(
        [row["p_raw"] for row in wilcoxon_rows]
    )

    for row, p_holm in zip(wilcoxon_rows, adjusted):
        row["p_holm"] = float(p_holm)

        row["significantly_better_than_3_layer"] = bool(
            row["mean_delta_auc"] > 0
            and row["p_holm"] < 0.05
        )


    print("\n" + "=" * 100)
    print(f"SUMMARY: {task['label']}")
    print("=" * 100)

    for row in summary_rows:
        print(
            f"{row['variant']:16s} | "
            f"params={row['parameters']:,} | "
            f"val AUROC={row['validation_auc_mean']*100:.2f} ± "
            f"{row['validation_auc_sd']*100:.2f}% | "
            f"test AUROC={row['test_auc_mean']*100:.2f} ± "
            f"{row['test_auc_sd']*100:.2f}%"
        )

    print("\nPAIRED TEST-AUROC COMPARISONS VS TWO-BLOCK CNN")
    print("-" * 100)

    for row in wilcoxon_rows:
        star = "*" if row["significantly_better_than_3_layer"] else ""

        print(
            f"{row['variant']:16s} | "
            f"mean ΔAUROC={row['mean_delta_auc']*100:+.2f} pp | "
            f"raw p={row['p_raw']:.6g} | "
            f"Holm p={row['p_holm']:.6g} {star}"
        )

    return results, summary_rows, wilcoxon_rows


all_results = {}
all_summary_rows = []
all_wilcoxon_rows = []

for task in TASKS:

    results, summary_rows, wilcoxon_rows = run_task(task)

    all_results[task["name"]] = results
    all_summary_rows.extend(summary_rows)
    all_wilcoxon_rows.extend(wilcoxon_rows)


# SAVE COMBINED SUMMARY CSV

summary_file = OUTPUT_DIR /"cnn_ablation_ALL_tasks_summary.csv"

with summary_file.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = list(all_summary_rows[0].keys())

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(all_summary_rows)


# SAVE COMBINED WILCOXON CSV

wilcoxon_file = OUTPUT_DIR / "cnn_ablation_ALL_tasks_wilcoxon_vs_3_layer.csv"

with wilcoxon_file.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = list(all_wilcoxon_rows[0].keys())

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(all_wilcoxon_rows)


# SAVE PER-REPETITION RESULTS FOR ALL TASKS

per_repeat_file = OUTPUT_DIR /  "cnn_ablation_ALL_tasks_per_repeat_results.csv"


with per_repeat_file.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "task",
            "task_label",
            "repeat",
            "variant",
            "validation_auc",
            "test_auc",
            "accuracy",
            "sensitivity",
            "specificity",
            "validation_threshold",
            "best_epoch",
            "n_parameters",
        ]
    )

    for task in TASKS:

        task_name = task["name"]
        task_label = task["label"]
        task_results = all_results[task_name]

        for variant in ablation_variants:

            for i in range(n_repeats):

                writer.writerow(
                    [
                        task_name,
                        task_label,
                        i + 1,
                        variant,
                        task_results[variant]["validation_auc"][i],
                        task_results[variant]["test_auc"][i],
                        task_results[variant]["accuracy"][i],
                        task_results[variant]["sensitivity"][i],
                        task_results[variant]["specificity"][i],
                        task_results[variant]["threshold"][i],
                        task_results[variant]["best_epoch"][i],
                        task_results[variant]["n_parameters"][i],
                    ]
                )


# SAVE HUMAN-READABLE COMBINED REPORT

report_file = OUTPUT_DIR / "cnn_ablation_ALL_tasks_results.txt"


with report_file.open(
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "CNN ARCHITECTURAL ABLATION STUDY - ALL TASKS\n"
    )
    f.write("=" * 80 + "\n\n")

    f.write(
        f"Reference architecture: {REFERENCE_VARIANT}\n"
    )
    f.write(
        f"Patient-independent repetitions per task: {n_repeats}\n"
    )
    f.write(
        f"Learning rate: {learning_rate}\n"
    )
    f.write(
        f"Weight decay: {weight_decay}\n"
    )
    f.write(
        f"Kernel size: {kernel_size}\n"
    )
    f.write(
        f"Dropout: {dropout}\n"
    )
    f.write(
        f"Batch size: {batch_size}\n"
    )
    f.write(
        f"Maximum epochs: {max_epochs}\n"
    )
    f.write(
        f"Early-stopping patience: "
        f"{early_stopping_patience}\n\n"
    )

    for task in TASKS:

        task_name = task["name"]

        f.write("\n" + "=" * 80 + "\n")
        f.write(task["label"] + "\n")
        f.write("=" * 80 + "\n\n")

        task_summary = [
            row
            for row in all_summary_rows
            if row["task"] == task_name
        ]

        for row in task_summary:

            f.write(
                f"{row['variant']}\n"
            )
            f.write(
                f"  parameters = "
                f"{row['parameters']:,}\n"
            )
            f.write(
                f"  validation AUROC = "
                f"{row['validation_auc_mean']*100:.2f} ± "
                f"{row['validation_auc_sd']*100:.2f}%\n"
            )
            f.write(
                f"  test AUROC = "
                f"{row['test_auc_mean']*100:.2f} ± "
                f"{row['test_auc_sd']*100:.2f}%\n"
            )
            f.write(
                f"  accuracy = "
                f"{row['accuracy_mean']*100:.2f} ± "
                f"{row['accuracy_sd']*100:.2f}%\n"
            )
            f.write(
                f"  sensitivity = "
                f"{row['sensitivity_mean']*100:.2f} ± "
                f"{row['sensitivity_sd']*100:.2f}%\n"
            )
            f.write(
                f"  specificity = "
                f"{row['specificity_mean']*100:.2f} ± "
                f"{row['specificity_sd']*100:.2f}%\n"
            )
            f.write(
                f"  best epoch = "
                f"{row['best_epoch_mean']:.1f} ± "
                f"{row['best_epoch_sd']:.1f}\n\n"
            )

        f.write(
            "PAIRED TEST-AUROC COMPARISONS VS TWO-BLOCK CNN\n"
        )
        f.write("-" * 80 + "\n")

        task_tests = [
            row
            for row in all_wilcoxon_rows
            if row["task"] == task_name
        ]

        for row in task_tests:

            star = (
                "*"
                if row["significantly_better_than_3_layer"]
                else ""
            )

            f.write(
                f"{row['variant']}: "
                f"mean ΔAUROC="
                f"{row['mean_delta_auc']*100:+.2f} percentage points, "
                f"raw p={row['p_raw']:.6g}, "
                f"Holm p={row['p_holm']:.6g}"
                f"{' *' if star else ''}\n"
            )


print("\n")
print("=" * 110)
print("COMBINED MANUSCRIPT TABLE")
print("'*' = significantly better than TWO-BLOCK CNN after Holm correction")
print("=" * 110)

header = (
    f"{'Architecture':18s} | {'Parameters':>10s}"
)

for task in TASKS:
    header += f" | {task['name']:^26s}"

print(header)
print("-" * len(header))

for variant in ablation_variants:

    parameter_row = next(
        row
        for row in all_summary_rows
        if row["variant"] == variant
    )

    line = (
        f"{variant:18s} | "
        f"{parameter_row['parameters']:10,d}"
    )

    for task in TASKS:

        summary_row = next(
            row
            for row in all_summary_rows
            if (
                row["task"] == task["name"]
                and row["variant"] == variant
            )
        )

        star = ""

        if variant != REFERENCE_VARIANT:

            test_row = next(
                row
                for row in all_wilcoxon_rows
                if (
                    row["task"] == task["name"]
                    and row["variant"] == variant
                )
            )

            if test_row["significantly_better_than_3_layer"]:
                star = "*"

        value = (
            f"{summary_row['test_auc_mean']*100:.2f} ± "
            f"{summary_row['test_auc_sd']*100:.2f}{star}"
        )

        line += f" | {value:^26s}"

    print(line)


print("\nSaved:")
print(" ", summary_file)
print(" ", wilcoxon_file)
print(" ", per_repeat_file)
print(" ", report_file)

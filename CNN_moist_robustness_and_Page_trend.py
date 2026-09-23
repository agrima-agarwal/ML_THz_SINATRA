# -*- coding: utf-8 -*-
"""
Hydration robustness experiment using the selected 3-layer CNN.

For each of 50 patient-independent repetitions:
    1. Use the same patients at all three conditions after applying the union blacklist.
    2. Split common patients: 70% development / 30% test.
    3. Split development patients: 80% train / 20% validation.
    4. Train the CNN only on no-moisturiser (0 min) data.
    5. Select the classification threshold from 0-min validation data using Youden's J.
    6. Freeze the model and evaluate the same held-out patients at:
           - 0 min
           - 10 min
           - 20 min
    7. Save metrics, predictions, ROC data, models and plots in output_2/.

CNN kept identical to the selected "3_layer" model:
    Conv1d(229 -> 64, kernel=3, padding=1) -> ReLU -> MaxPool1d(2)
    Conv1d(64 -> 128, kernel=3, padding=1) -> ReLU -> MaxPool1d(2)
    Flatten -> Linear(... -> 2)

Training settings kept identical:
    Adam, lr=1e-3, weight_decay=1e-4
    CrossEntropyLoss
    batch size=8
    max epochs=50
    early stopping patience=6, min_delta=1e-4
"""

import copy
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from scipy.stats import page_trend_test, rankdata

from functions import THzData


# ============================================================
# PLOT FORMATTING
# ============================================================

plt.rcdefaults()

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "legend.fontsize": 14,
    "lines.linewidth": 3,
    "axes.linewidth": 1.5,
    "savefig.dpi": 300,
})


# ============================================================
# SETTINGS
# ============================================================

RAW_ROOT = Path("raw files dry skin")
OUTPUT_DIR = Path("outputs/CNN_hydration_score")
DATASET_DIR = OUTPUT_DIR / "datasets"
SPLIT_DIR = OUTPUT_DIR / "splits"
MODEL_DIR = OUTPUT_DIR / "models"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
METRIC_DIR = OUTPUT_DIR / "metrics"
ROC_DIR = OUTPUT_DIR / "roc_data"
PLOT_DIR = OUTPUT_DIR / "plots"
PAGE_TEST_DIR = OUTPUT_DIR / "page_test"

for folder in [OUTPUT_DIR, DATASET_DIR, SPLIT_DIR, MODEL_DIR, PREDICTION_DIR, METRIC_DIR, ROC_DIR, PLOT_DIR, PAGE_TEST_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.30
VALIDATION_SIZE = 0.20
N_REPEATS = 50

BATCH_SIZE = 8
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 6
MIN_DELTA = 1e-4

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
KERNEL_SIZE = 3
DROPOUT = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNION_BLACK_LIST = {
    "S003", "S006", "S008", "S012", "S014", "S016",
    "S027", "S028", "S036", "S063", "S073",
}

CONDITIONS = {
    "no_moist": {
        "label": "No moisturiser",
        "roi_folder": "roi",
        "control_folder": "control",
        "minutes": 0,
    },
    "10_min": {
        "label": "10 min post",
        "roi_folder": "roi_10min",
        "control_folder": "control_10min",
        "minutes": 10,
    },
    "20_min": {
        "label": "20 min post",
        "roi_folder": "roi_20min",
        "control_folder": "control_20min",
        "minutes": 20,
    },
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# PATIENT / DATA LOADING
# ============================================================

def get_patients():
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"Could not find raw-data folder: {RAW_ROOT}")

    patients = sorted([
        patient_folder.name
        for patient_folder in RAW_ROOT.iterdir()
        if patient_folder.is_dir() and patient_folder.name not in UNION_BLACK_LIST
    ])

    print(f"\nPatients used in all three conditions: {len(patients)}")
    print(patients)

    pd.DataFrame({"patient_id": patients}).to_csv(
        OUTPUT_DIR / "patients_used.csv",
        index=False,
    )

    return patients

def load_measurement(patient_id, folder_name):
    patient_folder = RAW_ROOT / patient_id

    data = THzData(
        str(patient_folder / folder_name),
        str(patient_folder / "reference.txt"),
        str(patient_folder / "baseline.txt"),
    )

    return data.impulses.T.flatten()


def build_condition_dataset(patients, condition_name):
    condition = CONDITIONS[condition_name]

    roi_data = []
    control_data = []

    print(f"\nLoading {condition['label']}...")

    for i, patient_id in enumerate(patients, 1):
        print(f"  [{i:02d}/{len(patients):02d}] {patient_id}")

        roi_data.append(load_measurement(patient_id, condition["roi_folder"]))
        control_data.append(load_measurement(patient_id, condition["control_folder"]))

    X = np.asarray(roi_data + control_data)
    y = np.concatenate([
        np.ones(len(patients), dtype=int),
        np.zeros(len(patients), dtype=int),
    ])

    patient_ids = np.asarray(patients + patients)
    regions = np.asarray(
        ["roi"] * len(patients) +
        ["control"] * len(patients)
    )

    np.savez(
        DATASET_DIR / f"{condition_name}_common_patients.npz",
        X=X,
        y=y,
        patient_ids=patient_ids,
        region=regions,
        hydration_minutes=np.repeat(condition["minutes"], len(y)),
    )

    print(
        f"  Saved {condition_name}: X={X.shape}, y={y.shape}, "
        f"patients={len(patients)}"
    )

    return {
        "X": X,
        "y": y,
        "patient_ids": patient_ids,
        "region": regions,
        "minutes": condition["minutes"],
        "label": condition["label"],
    }


def prepare_cnn_X(X, y):
    X_cnn = X.reshape(len(y), 400, 229, order="F")
    X_cnn = np.transpose(X_cnn, (0, 2, 1))
    return torch.tensor(X_cnn, dtype=torch.float32)


# ============================================================
# SELECTED 3-LAYER CNN
# ============================================================

class ThreeLayerCNN1D(nn.Module):
    def __init__(self, input_channels=229, input_length=400):
        super().__init__()

        padding = KERNEL_SIZE // 2

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, KERNEL_SIZE, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, KERNEL_SIZE, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.dropout = nn.Dropout(DROPOUT) if DROPOUT > 0 else nn.Identity()

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_length)
            feature_shape = self.features(dummy).shape

        flattened_features = feature_shape[1] * feature_shape[2]
        self.fc = nn.Linear(flattened_features, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


# ============================================================
# EARLY STOPPING / TRAINING
# ============================================================

class EarlyStopping:
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_state = None
        self.best_epoch = 0
        self.counter = 0
        self.stop = False

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


def train_model(X_train, y_train, X_val, y_val, seed):
    set_all_seeds(seed)

    model = ThreeLayerCNN1D(
        input_channels=X_train.shape[1],
        input_length=X_train.shape[2],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    X_val_device = X_val.to(DEVICE)
    y_val_device = y_val.to(DEVICE)

    early_stopping = EarlyStopping()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            val_logits = model(X_val_device)
            val_loss = criterion(val_logits, y_val_device).item()

        early_stopping.step(val_loss, model, epoch)

        if early_stopping.stop:
            break

    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

    return model, early_stopping.best_epoch, early_stopping.best_loss


# ============================================================
# METRICS
# ============================================================

def predict_scores(model, X):
    model.eval()

    scores = []

    loader = DataLoader(
        TensorDataset(X),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(DEVICE))
            probs = torch.softmax(logits, dim=1)[:, 1]
            scores.extend(probs.cpu().numpy())

    return np.asarray(scores)


def youden_threshold(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    finite = np.isfinite(thresholds)
    fpr = fpr[finite]
    tpr = tpr[finite]
    thresholds = thresholds[finite]

    if len(thresholds) == 0:
        return 0.5

    return float(thresholds[np.argmax(tpr - fpr)])


def calculate_metrics(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, scores)

    return {
        "auc": float(auc),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


# ============================================================
# ROC STORAGE
# ============================================================

def interpolate_roc(y_true, scores, mean_fpr):
    fpr, tpr, _ = roc_curve(y_true, scores)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tpr[-1] = 1.0
    return interp_tpr


# ============================================================
# PLOTTING
# ============================================================

def plot_metric_vs_hydration(summary_df, metric, ylabel, filename):
    x = summary_df["minutes"].to_numpy()
    mean = summary_df[f"{metric}_mean"].to_numpy()
    sd = summary_df[f"{metric}_sd"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(x, mean, yerr=sd, marker="o", capsize=4, linewidth=1.8)
    ax.set_xlabel("Time after moisturiser application (min)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0, 10, 20])
    ax.set_xticklabels(["No moisturizer", "10 min", "20 min"])
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mean_roc(roc_store):
    fig, ax = plt.subplots(figsize=(6, 6))

    for condition_name, condition in CONDITIONS.items():
        mean_fpr = roc_store[condition_name]["mean_fpr"]
        tprs = np.asarray(roc_store[condition_name]["tprs"])
        aucs = np.asarray(roc_store[condition_name]["aucs"])

        mean_tpr = np.mean(tprs, axis=0)
        sd_tpr = np.std(tprs, axis=0)

        ax.plot(
            mean_fpr,
            mean_tpr,
            linewidth=2,
            label=f"{condition['label']} (AUROC {np.mean(aucs):.3f} ± {np.std(aucs):.3f})",
        )

        ax.fill_between(
            mean_fpr,
            np.clip(mean_tpr - sd_tpr, 0, 1),
            np.clip(mean_tpr + sd_tpr, 0, 1),
            alpha=0.15,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(framealpha=0.5)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "roc_common_patients_0_10_20.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dry_score_vs_hydration(dry_summary_df):
    x = dry_summary_df["minutes"].to_numpy()
    mean = dry_summary_df["score_mean"].to_numpy()
    sd = dry_summary_df["score_sd"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(x, mean, yerr=sd, marker="o", capsize=4, linewidth=1.8)
    ax.set_xlabel("Time after moisturiser application")
    ax.set_ylabel("CNN dry-class probability")
    ax.set_xticks([0, 10, 20])
    ax.set_xticklabels(["No moisturizer", "10 min", "20 min"])
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "dry_skin_score_vs_hydration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paired_dry_scores(dry_patient_mean_df):
    fig, ax = plt.subplots(figsize=(6, 5))

    for repeat in sorted(dry_patient_mean_df["repeat"].unique()):
        df_r = dry_patient_mean_df[dry_patient_mean_df["repeat"] == repeat]

        for patient_id in df_r["patient_id"].unique():
            df_p = df_r[df_r["patient_id"] == patient_id].sort_values("minutes")
            ax.plot(
                df_p["minutes"].to_numpy(),
                df_p["score_dry"].to_numpy(),
                marker="o",
                linewidth=0.7,
                alpha=0.15,
            )

    summary = dry_patient_mean_df.groupby("minutes")["score_dry"].agg(["mean", "std"]).reset_index()

    ax.errorbar(
        summary["minutes"],
        summary["mean"],
        yerr=summary["std"],
        marker="o",
        linewidth=2.5,
        capsize=4,
        label="Mean ± SD",
    )

    ax.set_xlabel("Time after moisturiser application (min)")
    ax.set_ylabel("CNN dry-class probability")
    ax.set_xticks([0, 10, 20])
    ax.set_ylim(0, 1)
    ax.legend(framealpha=0.5)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "dry_skin_paired_scores_vs_hydration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)



# ============================================================
# PAGE TREND ANALYSIS OF DRY-SKIN CNN SCORES
# ============================================================

def run_page_trend_analysis():

    file = METRIC_DIR / "dry_skin_paired_score_changes.csv"

    score_cols = [
        "score_0min",
        "score_10min",
        "score_20min",
    ]

    time_labels = [
        "No moisturizer",
        "10 min",
        "20 min",
    ]

    # Expected ordered decrease:
    # Baseline > 10 min > 20 min
    expected_ranks = [3, 2, 1]

    df = pd.read_csv(file)

    required_cols = [
        "repeat",
        "patient_id",
        *score_cols,
    ]

    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.dropna(
        subset=["patient_id", *score_cols]
    ).copy()

    print("\n" + "=" * 70)
    print("PAGE TREND ANALYSIS")
    print("=" * 70)

    print(
        f"Patient-repeat observations: "
        f"{len(df)}"
    )

    print(
        f"Unique patients: "
        f"{df['patient_id'].nunique()}"
    )

    print(
        f"Repeated splits: "
        f"{df['repeat'].nunique()}"
    )

    print(
        "\nNumber of test-set appearances per patient:"
    )

    print(
        df.groupby("patient_id")
          .size()
          .describe()
    )

    # --------------------------------------------------------
    # Primary analysis: average raw CNN probabilities across
    # repeated test-set appearances for each patient.
    # --------------------------------------------------------

    patient_scores = (
        df.groupby("patient_id")[score_cols]
          .mean()
          .sort_index()
    )

    X_raw = patient_scores.to_numpy()

    page_raw = page_trend_test(
        X_raw,
        predicted_ranks=expected_ranks,
        method="auto",
    )

    print("\nPRIMARY ANALYSIS: RAW CNN SCORES")
    print(
        f"Number of patients: "
        f"{len(patient_scores)}"
    )

    print(
        f"Page L statistic: "
        f"{page_raw.statistic:.3f}"
    )

    print(
        f"One-sided p-value: "
        f"{page_raw.pvalue:.6f}"
    )

    print("\nPatient-level CNN scores:")

    for j, label in enumerate(time_labels):
        print(
            f"{label:10s} "
            f"mean = {X_raw[:, j].mean():.4f}, "
            f"median = {np.median(X_raw[:, j]):.4f}"
        )

    # --------------------------------------------------------
    # Plot raw patient-level scores.
    # --------------------------------------------------------

    x = np.arange(3)
    labels = ["No moisturizer", "10 min", "20 min"]

    raw_mean = X_raw.mean(axis=0)
    raw_sem = X_raw.std(axis=0, ddof=1) / np.sqrt(X_raw.shape[0])
    raw_ci95 = 1.96 * raw_sem

    fig, ax = plt.subplots(figsize=(5.5, 6))

    ax.errorbar(
        x,
        raw_mean,
        yerr=raw_ci95,
        marker="o",
        markersize=8,
        linewidth=3,
        capsize=5,
        color="blue",
        label="Mean ± 95% CI",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Time after moisturisation")
    ax.set_ylabel("CNN dry-class probability")

    ax.legend(
        loc="upper right",
        fontsize=18,
        framealpha=0.5,
    )

    plt.tight_layout()

    plt.savefig(
        PAGE_TEST_DIR / "dryness_raw_scores_page_test.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Calibration-invariant sensitivity analysis.
    #
    # Rank the three scores within each patient x repeat.
    # --------------------------------------------------------

    def within_row_ranks(row):

        values = row[
            score_cols
        ].to_numpy(
            dtype=float
        )

        return rankdata(
            values,
            method="average",
        )

    rank_array = np.vstack(
        df.apply(
            within_row_ranks,
            axis=1,
        )
    )

    rank_cols = [
        "rank_0min",
        "rank_10min",
        "rank_20min",
    ]

    df[rank_cols] = rank_array

    patient_ranks = (
        df.groupby("patient_id")[rank_cols]
          .mean()
          .sort_index()
    )

    X_rank = patient_ranks.to_numpy()

    page_rank = page_trend_test(
        X_rank,
        predicted_ranks=expected_ranks,
        method="auto",
    )

    print(
        "\nCALIBRATION-INVARIANT "
        "WITHIN-REPEAT RANK ANALYSIS"
    )

    for j, label in enumerate(time_labels):
        print(
            f"{label:10s} "
            f"mean rank = "
            f"{X_rank[:, j].mean():.4f}, "
            f"median rank = "
            f"{np.median(X_rank[:, j]):.4f}"
        )

    print(
        f"\nPage L statistic: "
        f"{page_rank.statistic:.3f}"
    )

    print(
        f"One-sided p-value: "
        f"{page_rank.pvalue:.6f}"
    )

    print(
        "\nSame Page statistic as "
        "raw-score analysis: "
        f"{np.isclose(page_raw.statistic, page_rank.statistic)}"
    )

    print(
        "Same p-value as "
        "raw-score analysis: "
        f"{np.isclose(page_raw.pvalue, page_rank.pvalue)}"
    )

    # --------------------------------------------------------
    # Plot within-repeat ranks.
    # --------------------------------------------------------

    rank_mean = X_rank.mean(axis=0)
    rank_sem = X_rank.std(axis=0, ddof=1) / np.sqrt(X_rank.shape[0])
    rank_ci95 = 1.96 * rank_sem

    fig, ax = plt.subplots(figsize=(5.5, 6))

    ax.errorbar(
        x,
        rank_mean,
        yerr=rank_ci95,
        marker="o",
        markersize=8,
        linewidth=3,
        capsize=5,
        color="blue",
        label="Mean rank ± 95% CI",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Time after moisturisation")
    ax.set_ylabel("Average within-run CNN rank")

    ax.set_ylim(0.8, 3.2)
    ax.set_yticks([1, 1.5, 2, 2.5, 3])

    ax.legend(
        loc="upper right",
        fontsize=18,
        framealpha=0.5,
    )

    plt.tight_layout()

    plt.savefig(
        PAGE_TEST_DIR / "dryness_rank_page_test.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Save patient-level values and rank data.
    # --------------------------------------------------------

    patient_results = (
        patient_scores
        .join(
            patient_ranks
        )
    )

    patient_results.to_csv(
        PAGE_TEST_DIR /
        "dryness_patient_level_results.csv"
    )

    df.to_csv(
        PAGE_TEST_DIR /
        "dryness_within_bootstrap_ranks.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save concise summary.
    # --------------------------------------------------------

    page_summary_df = pd.DataFrame([
        {
            "analysis":
                "raw_scores",
            "n_patients":
                len(
                    patient_scores
                ),
            "page_L":
                page_raw.statistic,
            "p_one_sided":
                page_raw.pvalue,
            "baseline_mean":
                raw_mean[0],
            "10min_mean":
                raw_mean[1],
            "20min_mean":
                raw_mean[2],
        },
        {
            "analysis":
                "within_run_ranks",
            "n_patients":
                len(
                    patient_ranks
                ),
            "page_L":
                page_rank.statistic,
            "p_one_sided":
                page_rank.pvalue,
            "baseline_mean":
                rank_mean[0],
            "10min_mean":
                rank_mean[1],
            "20min_mean":
                rank_mean[2],
        },
    ])

    page_summary_df.to_csv(
        PAGE_TEST_DIR /
        "dryness_page_test_summary.csv",
        index=False,
    )

    print("\nPAGE TEST SUMMARY")

    print(
        f"  Raw-score Page L = "
        f"{page_raw.statistic:.0f}, "
        f"p = "
        f"{page_raw.pvalue:.6f}"
    )

    print(
        f"  Rank Page L = "
        f"{page_rank.statistic:.0f}, "
        f"p = "
        f"{page_rank.pvalue:.6f}"
    )

    print(
        "\nPage-test outputs saved under:"
    )

    print(
        PAGE_TEST_DIR.resolve()
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    print("\n" + "=" * 90)
    print("HYDRATION ROBUSTNESS: TRAIN 0 MIN -> TEST 0 / 10 / 20 MIN")
    print("=" * 90)
    print("Device:", DEVICE)

    patients = get_patients()

    datasets = {
        condition_name: build_condition_dataset(patients, condition_name)
        for condition_name in CONDITIONS
    }

    n_patients = len(patients)
    unique_patients = np.arange(n_patients)

    for condition_name in CONDITIONS:
        datasets[condition_name]["X_cnn"] = prepare_cnn_X(
            datasets[condition_name]["X"],
            datasets[condition_name]["y"],
        )

    # Every condition is built in the same order:
    # first all ROI samples, then all control samples.
    sample_patient_indices = np.concatenate([
        unique_patients,
        unique_patients,
    ])

    expected_y = datasets["no_moist"]["y"]

    for condition_name in CONDITIONS:
        if not np.array_equal(datasets[condition_name]["y"], expected_y):
            raise ValueError(f"Label ordering mismatch in {condition_name}.")

        if not np.array_equal(
            datasets[condition_name]["patient_ids"],
            datasets["no_moist"]["patient_ids"],
        ):
            raise ValueError(f"Patient ordering mismatch in {condition_name}.")

    metric_rows = []
    prediction_rows = []
    split_rows = []
    dry_score_rows = []

    mean_fpr = np.linspace(0, 1, 201)

    roc_store = {
        condition_name: {
            "mean_fpr": mean_fpr,
            "tprs": [],
            "aucs": [],
        }
        for condition_name in CONDITIONS
    }

    for repeat_i in range(N_REPEATS):
        print("\n" + "-" * 90)
        print(f"Repetition {repeat_i + 1}/{N_REPEATS}")
        print("-" * 90)

        train_pool_patients, test_patients = train_test_split(
            unique_patients,
            test_size=TEST_SIZE,
            random_state=repeat_i,
        )

        train_patients, val_patients = train_test_split(
            train_pool_patients,
            test_size=VALIDATION_SIZE,
            random_state=repeat_i,
        )

        train_mask = np.isin(sample_patient_indices, train_patients)
        val_mask = np.isin(sample_patient_indices, val_patients)
        test_mask = np.isin(sample_patient_indices, test_patients)

        for split_name, patient_indices in [
            ("train", train_patients),
            ("validation", val_patients),
            ("test", test_patients),
        ]:
            for patient_index in patient_indices:
                split_rows.append({
                    "repeat": repeat_i + 1,
                    "split": split_name,
                    "patient_index": int(patient_index),
                    "patient_id": patients[int(patient_index)],
                })

        X_train = datasets["no_moist"]["X_cnn"][train_mask]
        y_train = torch.tensor(expected_y[train_mask], dtype=torch.long)

        X_val = datasets["no_moist"]["X_cnn"][val_mask]
        y_val_np = expected_y[val_mask]
        y_val = torch.tensor(y_val_np, dtype=torch.long)

        training_seed = 1000 + repeat_i

        model, best_epoch, best_val_loss = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            training_seed,
        )

        val_scores = predict_scores(model, X_val)
        val_auc = roc_auc_score(y_val_np, val_scores)
        threshold = youden_threshold(y_val_np, val_scores)

        model_path = MODEL_DIR / f"repeat_{repeat_i + 1:02d}.pt"

        torch.save({
            "model_state_dict": model.state_dict(),
            "repeat": repeat_i + 1,
            "seed": training_seed,
            "train_patients": np.asarray([patients[i] for i in train_patients]),
            "validation_patients": np.asarray([patients[i] for i in val_patients]),
            "test_patients": np.asarray([patients[i] for i in test_patients]),
            "threshold": threshold,
            "validation_auc": val_auc,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "input_channels": 229,
            "input_length": 400,
        }, model_path)

        print(
            f"Train patients={len(train_patients)} | "
            f"Val patients={len(val_patients)} | "
            f"Test patients={len(test_patients)} | "
            f"best epoch={best_epoch} | "
            f"val AUROC={val_auc:.4f} | "
            f"threshold={threshold:.4f}"
        )

        for condition_name, condition in CONDITIONS.items():
            X_test = datasets[condition_name]["X_cnn"][test_mask]
            y_test = datasets[condition_name]["y"][test_mask]
            patient_ids_test = datasets[condition_name]["patient_ids"][test_mask]
            regions_test = datasets[condition_name]["region"][test_mask]

            scores = predict_scores(model, X_test)
            metrics = calculate_metrics(y_test, scores, threshold)

            interp_tpr = interpolate_roc(y_test, scores, mean_fpr)
            roc_store[condition_name]["tprs"].append(interp_tpr)
            roc_store[condition_name]["aucs"].append(metrics["auc"])

            metric_rows.append({
                "repeat": repeat_i + 1,
                "condition": condition_name,
                "condition_label": condition["label"],
                "minutes": condition["minutes"],
                "validation_auc_0min": val_auc,
                "threshold_from_0min_validation": threshold,
                "best_epoch": best_epoch,
                "auc": metrics["auc"],
                "accuracy": metrics["accuracy"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "n_test_patients": len(test_patients),
                "n_test_samples": len(y_test),
            })

            for patient_id, region, true_label, score in zip(
                patient_ids_test,
                regions_test,
                y_test,
                scores,
            ):
                prediction_rows.append({
                    "repeat": repeat_i + 1,
                    "condition": condition_name,
                    "condition_label": condition["label"],
                    "minutes": condition["minutes"],
                    "patient_id": patient_id,
                    "region": region,
                    "true_label": int(true_label),
                    "score_dry": float(score),
                    "predicted_label": int(score >= threshold),
                    "threshold_from_0min_validation": threshold,
                })

                # Dry-skin score analysis: ROI samples only.
                if region == "roi":
                    dry_score_rows.append({
                        "repeat": repeat_i + 1,
                        "condition": condition_name,
                        "condition_label": condition["label"],
                        "minutes": condition["minutes"],
                        "patient_id": patient_id,
                        "score_dry": float(score),
                        "threshold_from_0min_validation": threshold,
                    })

            print(
                f"  {condition['label']:16s} | "
                f"AUROC={metrics['auc']:.4f} | "
                f"Acc={metrics['accuracy']:.4f} | "
                f"Sens={metrics['sensitivity']:.4f} | "
                f"Spec={metrics['specificity']:.4f}"
            )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    splits_df = pd.DataFrame(split_rows)
    dry_scores_df = pd.DataFrame(dry_score_rows)

    metrics_df.to_csv(METRIC_DIR / "per_repeat_metrics.csv", index=False)
    predictions_df.to_csv(PREDICTION_DIR / "all_test_predictions.csv", index=False)
    splits_df.to_csv(SPLIT_DIR / "patient_splits.csv", index=False)
    dry_scores_df.to_csv(PREDICTION_DIR / "dry_skin_scores_all_repeats.csv", index=False)

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows = []

    for condition_name, condition in CONDITIONS.items():
        df = metrics_df[metrics_df["condition"] == condition_name]

        row = {
            "condition": condition_name,
            "condition_label": condition["label"],
            "minutes": condition["minutes"],
        }

        for metric in ["auc", "accuracy", "sensitivity", "specificity"]:
            row[f"{metric}_mean"] = df[metric].mean()
            row[f"{metric}_sd"] = df[metric].std(ddof=0)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("minutes")
    summary_df.to_csv(METRIC_DIR / "summary_metrics.csv", index=False)

    # Paired changes relative to the no-moisturiser test score.
    pivot_auc = metrics_df.pivot(
        index="repeat",
        columns="condition",
        values="auc",
    )

    paired_change_df = pd.DataFrame({
        "repeat": pivot_auc.index,
        "auc_no_moist": pivot_auc["no_moist"].values,
        "auc_10_min": pivot_auc["10_min"].values,
        "auc_20_min": pivot_auc["20_min"].values,
        "delta_auc_10_minus_0": (
            pivot_auc["10_min"] - pivot_auc["no_moist"]
        ).values,
        "delta_auc_20_minus_0": (
            pivot_auc["20_min"] - pivot_auc["no_moist"]
        ).values,
    })

    paired_change_df.to_csv(
        METRIC_DIR / "paired_auc_changes.csv",
        index=False,
    )

    # ========================================================
    # DRY-SKIN-ONLY SCORE CHANGE
    # ========================================================

    dry_summary_df = (
        dry_scores_df.groupby(["condition", "condition_label", "minutes"])["score_dry"]
        .agg(score_mean="mean", score_sd="std", n="count")
        .reset_index()
        .sort_values("minutes")
    )

    dry_summary_df.to_csv(
        METRIC_DIR / "dry_skin_score_summary.csv",
        index=False,
    )

    dry_wide_df = dry_scores_df.pivot_table(
        index=["repeat", "patient_id"],
        columns="minutes",
        values="score_dry",
        aggfunc="mean",
    ).reset_index()

    dry_wide_df = dry_wide_df.rename(columns={
        0: "score_0min",
        10: "score_10min",
        20: "score_20min",
    })

    dry_wide_df["delta_10_minus_0"] = (
        dry_wide_df["score_10min"] - dry_wide_df["score_0min"]
    )
    dry_wide_df["delta_20_minus_0"] = (
        dry_wide_df["score_20min"] - dry_wide_df["score_0min"]
    )
    dry_wide_df["delta_20_minus_10"] = (
        dry_wide_df["score_20min"] - dry_wide_df["score_10min"]
    )

    dry_wide_df.to_csv(
        METRIC_DIR / "dry_skin_paired_score_changes.csv",
        index=False,
    )

    # One row per repeat/patient/condition, useful for paired plotting.
    dry_patient_mean_df = (
        dry_scores_df.groupby(["repeat", "patient_id", "minutes"], as_index=False)["score_dry"]
        .mean()
    )

    # ========================================================
    # SAVE ROC ARRAYS
    # ========================================================

    for condition_name, condition in CONDITIONS.items():
        tprs = np.asarray(roc_store[condition_name]["tprs"])
        aucs = np.asarray(roc_store[condition_name]["aucs"])

        np.savez(
            ROC_DIR / f"{condition_name}_roc.npz",
            mean_fpr=mean_fpr,
            tprs=tprs,
            mean_tpr=np.mean(tprs, axis=0),
            sd_tpr=np.std(tprs, axis=0),
            aucs=aucs,
            auc_mean=np.mean(aucs),
            auc_sd=np.std(aucs),
        )

        pd.DataFrame({
            "fpr": mean_fpr,
            "mean_tpr": np.mean(tprs, axis=0),
            "sd_tpr": np.std(tprs, axis=0),
        }).to_csv(
            ROC_DIR / f"{condition_name}_mean_roc.csv",
            index=False,
        )

    # ========================================================
    # PLOTS
    # ========================================================

    plot_mean_roc(roc_store)

    plot_metric_vs_hydration(
        summary_df,
        "auc",
        "AUROC",
        "auroc_vs_hydration.png",
    )

    plot_metric_vs_hydration(
        summary_df,
        "accuracy",
        "Accuracy",
        "accuracy_vs_hydration.png",
    )

    plot_metric_vs_hydration(
        summary_df,
        "sensitivity",
        "Sensitivity",
        "sensitivity_vs_hydration.png",
    )

    plot_metric_vs_hydration(
        summary_df,
        "specificity",
        "Specificity",
        "specificity_vs_hydration.png",
    )

    plot_dry_score_vs_hydration(dry_summary_df)
    plot_paired_dry_scores(dry_patient_mean_df)

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)

    for _, row in summary_df.iterrows():
        print(
            f"{row['condition_label']:16s} | "
            f"AUROC {row['auc_mean']:.4f} ± {row['auc_sd']:.4f} | "
            f"Acc {row['accuracy_mean']:.4f} ± {row['accuracy_sd']:.4f} | "
            f"Sens {row['sensitivity_mean']:.4f} ± {row['sensitivity_sd']:.4f} | "
            f"Spec {row['specificity_mean']:.4f} ± {row['specificity_sd']:.4f}"
        )

    print("\nMean paired AUROC changes:")
    print(
        "  10 min - 0 min:",
        f"{paired_change_df['delta_auc_10_minus_0'].mean():.4f} ± "
        f"{paired_change_df['delta_auc_10_minus_0'].std(ddof=0):.4f}",
    )
    print(
        "  20 min - 0 min:",
        f"{paired_change_df['delta_auc_20_minus_0'].mean():.4f} ± "
        f"{paired_change_df['delta_auc_20_minus_0'].std(ddof=0):.4f}",
    )

    print("\nDry-skin CNN score summary:")
    for _, row in dry_summary_df.iterrows():
        print(
            f"  {row['condition_label']:16s} | "
            f"score {row['score_mean']:.4f} ± {row['score_sd']:.4f}"
        )

    print("\nMean paired dry-score changes:")
    print(
        "  10 min - 0 min:",
        f"{dry_wide_df['delta_10_minus_0'].mean():.4f} ± "
        f"{dry_wide_df['delta_10_minus_0'].std(ddof=0):.4f}",
    )
    print(
        "  20 min - 0 min:",
        f"{dry_wide_df['delta_20_minus_0'].mean():.4f} ± "
        f"{dry_wide_df['delta_20_minus_0'].std(ddof=0):.4f}",
    )

    print("\nSaved everything under:", OUTPUT_DIR.resolve())

    # Run Page trend analysis automatically on the saved
    # dry-skin paired CNN scores.
    run_page_trend_analysis()


if __name__ == "__main__":
    main()

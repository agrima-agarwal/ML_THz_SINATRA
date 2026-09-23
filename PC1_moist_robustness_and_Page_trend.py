# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
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

RAW_ROOT = "raw files dry skin"
OUTPUT_DIR = "outputs/PC1_hydration_score"
N_REPEATS = 50
TEST_SIZE = 0.30
VALIDATION_SIZE = 0.20

UNION_BLACK_LIST = {
    "S003", "S006", "S008", "S012", "S014", "S016",
    "S027", "S028", "S036", "S063", "S073",
}

CONDITIONS = {
    "no_moist": {"label": "No moisturizer", "roi_folder": "roi", "control_folder": "control", "minutes": 0},
    "10_min": {"label": "10 min", "roi_folder": "roi_10min", "control_folder": "control_10min", "minutes": 10},
    "20_min": {"label": "20 min", "roi_folder": "roi_20min", "control_folder": "control_20min", "minutes": 20},
}

DATASET_DIR = os.path.join(OUTPUT_DIR, "datasets")
SPLIT_DIR = os.path.join(OUTPUT_DIR, "splits")
PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions")
METRIC_DIR = os.path.join(OUTPUT_DIR, "metrics")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
PAGE_TEST_DIR = os.path.join(OUTPUT_DIR, "page_test")

for folder in [OUTPUT_DIR, DATASET_DIR, SPLIT_DIR, PREDICTION_DIR, METRIC_DIR, PLOT_DIR, PAGE_TEST_DIR]:
    os.makedirs(folder, exist_ok=True)


def get_patients():
    if not os.path.isdir(RAW_ROOT):
        raise FileNotFoundError(f"Could not find raw-data folder: {RAW_ROOT}")

    patients = sorted([
        patient_id for patient_id in os.listdir(RAW_ROOT)
        if os.path.isdir(os.path.join(RAW_ROOT, patient_id)) and patient_id not in UNION_BLACK_LIST
    ])

    print(f"\nPatients used in all conditions: {len(patients)}")
    print(patients)
    pd.DataFrame({"patient_id": patients}).to_csv(os.path.join(OUTPUT_DIR, "patients_used.csv"), index=False)
    return patients


def load_measurement(patient_id, folder_name):
    patient_folder = os.path.join(RAW_ROOT, patient_id)
    data = THzData(
        os.path.join(patient_folder, folder_name),
        os.path.join(patient_folder, "reference.txt"),
        os.path.join(patient_folder, "baseline.txt"),
    )
    return data.impulses.T.flatten()


def build_condition_dataset(patients, condition_name):
    condition = CONDITIONS[condition_name]
    roi_data, control_data = [], []

    print(f"\nLoading {condition['label']}...")
    for i, patient_id in enumerate(patients, 1):
        print(f"  [{i:02d}/{len(patients):02d}] {patient_id}")
        roi_data.append(load_measurement(patient_id, condition["roi_folder"]))
        control_data.append(load_measurement(patient_id, condition["control_folder"]))

    X = np.asarray(roi_data + control_data)
    y = np.concatenate([np.ones(len(patients), dtype=int), np.zeros(len(patients), dtype=int)])
    patient_ids = np.asarray(patients + patients)
    regions = np.asarray(["roi"] * len(patients) + ["control"] * len(patients))

    np.savez(
        os.path.join(DATASET_DIR, f"{condition_name}.npz"),
        X=X, y=y, patient_ids=patient_ids, region=regions,
        hydration_minutes=np.repeat(condition["minutes"], len(y)),
    )

    return {"X": X, "y": y, "patient_ids": patient_ids, "region": regions,
            "minutes": condition["minutes"], "label": condition["label"]}


# ============================================================
# PC1 MODEL: SAME LOGIC AS EXISTING PCA_model
# ============================================================

def fit_pc1(X_train, y_train):
    pca = PCA(n_components=1, random_state=42)
    pca.fit(X_train)
    train_score = pca.transform(X_train)[:, 0]

    sign = 1.0
    if np.mean(train_score[y_train == 1]) < np.mean(train_score[y_train == 0]):
        sign = -1.0

    return pca, sign


def transform_pc1(pca, sign, X):
    return sign * pca.transform(X)[:, 0]


def main():
    print("\n" + "=" * 90)
    print("PC1 HYDRATION ROBUSTNESS: TRAIN BASELINE -> TEST BASELINE / 10 / 20 MIN")
    print("=" * 90)

    patients = get_patients()
    datasets = {name: build_condition_dataset(patients, name) for name in CONDITIONS}

    n_patients = len(patients)
    unique_patients = np.arange(n_patients)
    sample_patient_indices = np.concatenate([unique_patients, unique_patients])

    for name in CONDITIONS:
        if not np.array_equal(datasets[name]["patient_ids"], datasets["no_moist"]["patient_ids"]):
            raise ValueError(f"Patient order mismatch in {name}.")

    prediction_rows, metric_rows, split_rows = [], [], []

    for repeat_i in range(N_REPEATS):
        print(f"\nRepeat {repeat_i + 1}/{N_REPEATS}")
        random.seed(repeat_i)
        np.random.seed(repeat_i)

        train_pool_patients, test_patients = train_test_split(
            unique_patients, test_size=TEST_SIZE, random_state=repeat_i
        )
        train_patients, validation_patients = train_test_split(
            train_pool_patients, test_size=VALIDATION_SIZE, random_state=repeat_i
        )

        train_mask = np.isin(sample_patient_indices, train_patients)
        test_mask = np.isin(sample_patient_indices, test_patients)

        for split_name, indices in [("train", train_patients), ("validation", validation_patients), ("test", test_patients)]:
            for patient_index in indices:
                split_rows.append({
                    "repeat": repeat_i + 1,
                    "split": split_name,
                    "patient_id": patients[patient_index],
                })

        # Fit PC1 only on no-moisturiser training samples.
        X_train = datasets["no_moist"]["X"][train_mask]
        y_train = datasets["no_moist"]["y"][train_mask]
        pca, sign = fit_pc1(X_train, y_train)

        # Apply the same frozen PCA transform to the same held-out patients at all hydration levels.
        for condition_name, condition in CONDITIONS.items():
            X_test = datasets[condition_name]["X"][test_mask]
            y_test = datasets[condition_name]["y"][test_mask]
            patient_ids_test = datasets[condition_name]["patient_ids"][test_mask]
            regions_test = datasets[condition_name]["region"][test_mask]

            pc1_scores = transform_pc1(pca, sign, X_test)
            auc = roc_auc_score(y_test, pc1_scores)

            metric_rows.append({
                "repeat": repeat_i + 1,
                "condition": condition_name,
                "minutes": condition["minutes"],
                "auc": auc,
                "explained_variance_ratio_pc1": pca.explained_variance_ratio_[0],
                "pc1_sign": sign,
                "n_test_patients": len(test_patients),
            })

            for patient_id, region, true_label, score in zip(
                patient_ids_test, regions_test, y_test, pc1_scores
            ):
                prediction_rows.append({
                    "repeat": repeat_i + 1,
                    "condition": condition_name,
                    "minutes": condition["minutes"],
                    "patient_id": patient_id,
                    "region": region,
                    "true_label": int(true_label),
                    "pc1_score": float(score),
                })

            print(f"  {condition['label']:8s} AUROC = {auc:.4f}")

    predictions_df = pd.DataFrame(prediction_rows)
    metrics_df = pd.DataFrame(metric_rows)
    splits_df = pd.DataFrame(split_rows)

    predictions_df.to_csv(os.path.join(PREDICTION_DIR, "pc1_all_test_predictions.csv"), index=False)
    metrics_df.to_csv(os.path.join(METRIC_DIR, "pc1_per_repeat_auc.csv"), index=False)
    splits_df.to_csv(os.path.join(SPLIT_DIR, "pc1_patient_splits.csv"), index=False)

    auc_summary = (
        metrics_df.groupby("minutes", as_index=False)
        .agg(mean_auc=("auc", "mean"), sd_auc=("auc", "std"), n_runs=("repeat", "nunique"))
        .sort_values("minutes")
    )
    auc_summary.to_csv(os.path.join(METRIC_DIR, "pc1_auc_summary.csv"), index=False)

    # Dry ROI scores only, same output structure as CNN paired-score file.
    dry_df = predictions_df[predictions_df["region"] == "roi"].copy()
    dry_wide = dry_df.pivot_table(
        index=["repeat", "patient_id"], columns="minutes", values="pc1_score", aggfunc="mean"
    ).reset_index()

    dry_wide = dry_wide.rename(columns={0: "score_0min", 10: "score_10min", 20: "score_20min"})
    dry_wide = dry_wide.dropna(subset=["score_0min", "score_10min", "score_20min"])
    dry_wide["delta_10_minus_0"] = dry_wide["score_10min"] - dry_wide["score_0min"]
    dry_wide["delta_20_minus_0"] = dry_wide["score_20min"] - dry_wide["score_0min"]
    dry_wide["delta_20_minus_10"] = dry_wide["score_20min"] - dry_wide["score_10min"]
    dry_wide.to_csv(os.path.join(METRIC_DIR, "pc1_dry_skin_paired_score_changes.csv"), index=False)

    # Patient-level primary analysis: average repeated held-out appearances first.
    score_cols = ["score_0min", "score_10min", "score_20min"]
    patient_scores = dry_wide.groupby("patient_id")[score_cols].mean().sort_index()
    X_raw = patient_scores.to_numpy()

    expected_ranks = [3, 2, 1]  # Baseline > 10 min > 20 min
    page_raw = page_trend_test(X_raw, predicted_ranks=expected_ranks, method="auto")

    # Calibration-invariant within-run rank sensitivity analysis.
    rank_array = np.vstack([
        rankdata(row, method="average")
        for row in dry_wide[score_cols].to_numpy(dtype=float)
    ])
    rank_cols = ["rank_0min", "rank_10min", "rank_20min"]
    dry_wide[rank_cols] = rank_array

    patient_ranks = dry_wide.groupby("patient_id")[rank_cols].mean().sort_index()
    X_rank = patient_ranks.to_numpy()
    page_rank = page_trend_test(X_rank, predicted_ranks=expected_ranks, method="auto")

    patient_results = patient_scores.join(patient_ranks)
    patient_results.to_csv(os.path.join(PAGE_TEST_DIR, "pc1_patient_level_results.csv"))
    dry_wide.to_csv(os.path.join(PAGE_TEST_DIR, "pc1_within_run_scores_and_ranks.csv"), index=False)

    page_summary = pd.DataFrame([
        {
            "analysis": "raw patient-level PC1 scores",
            "n_patients": len(patient_scores),
            "page_L": page_raw.statistic,
            "p_one_sided": page_raw.pvalue,
        },
        {
            "analysis": "within-run PC1 ranks",
            "n_patients": len(patient_ranks),
            "page_L": page_rank.statistic,
            "p_one_sided": page_rank.pvalue,
        },
    ])
    page_summary.to_csv(os.path.join(PAGE_TEST_DIR, "pc1_page_test_summary.csv"), index=False)

    # Plot raw patient-level PC1 scores.
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
    ax.set_ylabel("PC1 dry-skin score")

    ax.legend(
        loc="upper right",
        fontsize=18,
        framealpha=0.5,
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(PLOT_DIR, "pc1_dry_scores_page_test.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Plot within-run ranks.
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
    ax.set_ylabel("Average within-run PC1 rank")

    ax.set_ylim(0.8, 3.2)
    ax.set_yticks([1, 1.5, 2, 2.5, 3])

    ax.legend(
        loc="upper right",
        fontsize=18,
        framealpha=0.5,
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(PLOT_DIR, "pc1_dry_ranks_page_test.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\n" + "=" * 90)
    print("FINAL PC1 SUMMARY")
    print("=" * 90)
    print("\nAUROC:")
    print(auc_summary.to_string(index=False, float_format=lambda z: f"{z:.4f}"))

    print("\nPatient-level mean PC1 scores:")
    for label, col in zip(labels, score_cols):
        print(f"  {label:8s}: {patient_scores[col].mean():.4f}")

    print("\nPrimary Page trend test:")
    print(f"  L = {page_raw.statistic:.3f}")
    print(f"  p = {page_raw.pvalue:.6g}")

    print("\nRank-based sensitivity analysis:")
    print(f"  L = {page_rank.statistic:.3f}")
    print(f"  p = {page_rank.pvalue:.6g}")

    # Save concise Page-test summary, matching the CNN analysis.
    summary_df = pd.DataFrame([
        {
            "analysis": "raw_scores",
            "n_patients": len(patient_scores),
            "page_L": page_raw.statistic,
            "p_one_sided": page_raw.pvalue,
            "baseline_mean": raw_mean[0],
            "10min_mean": raw_mean[1],
            "20min_mean": raw_mean[2],
        },
        {
            "analysis": "within_run_ranks",
            "n_patients": len(patient_ranks),
            "page_L": page_rank.statistic,
            "p_one_sided": page_rank.pvalue,
            "baseline_mean": rank_mean[0],
            "10min_mean": rank_mean[1],
            "20min_mean": rank_mean[2],
        },
    ])

    summary_path = os.path.join(
        PAGE_TEST_DIR,
        "pc1_page_test_summary_detailed.csv",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print("\nSaved Page-test summary:")
    print(summary_path)

    print("\nPage-test outputs saved under:")
    print(os.path.abspath(PAGE_TEST_DIR))

    print("\nSaved everything under:")
    print(os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()

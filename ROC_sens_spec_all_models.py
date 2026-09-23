# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


INPUT_DIR = Path("outputs/training_results")
OUTPUT_DIR = Path("outputs/roc_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_FILES = [
    "0 all_tasks_roc_data.npz",
    "1 all_tasks_roc_data.npz",
    "2 all_tasks_roc_data.npz",
    "3 all_tasks_roc_data.npz",
    "4 all_tasks_roc_data.npz",
]

MODEL_NAMES = [
    "PC1",
    "LDA",
    "Logistic Regression",
    "Logistic Regression + PCA",
    "kNN",
    "kNN + PCA",
    "Decision Tree",
    "Random Forest",
    "SLP",
    "MLP",
    "CNN",
]

MODEL_LABELS = {
    "PC1": "PC1",
    "LDA": "LDA",
    "Logistic Regression": "LR",
    "Logistic Regression + PCA": "LR + PCA",
    "kNN": "kNN",
    "kNN + PCA": "kNN + PCA",
    "Decision Tree": "DT",
    "Random Forest": "RF",
    "SLP": "SLP",
    "MLP": "MLP",
    "CNN": "CNN",
}

plt.rcdefaults()
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "legend.fontsize": 9,
    "lines.linewidth": 2,
    "axes.linewidth": 1.5,
    "savefig.dpi": 300,
})


def safe_model_name(name):
    return name.replace(" ", "_").replace("+", "plus")


def load_model_repeats(data, model_name):
    safe_name = safe_model_name(model_name)
    y_all, scores_all = [], []
    repeat_i = 0

    while True:
        y_key = f"{safe_name}__repeat_{repeat_i:02d}__y_test"
        s_key = f"{safe_name}__repeat_{repeat_i:02d}__scores"
        if y_key not in data.files or s_key not in data.files:
            break
        y_all.append(np.asarray(data[y_key], dtype=int))
        scores_all.append(np.asarray(data[s_key], dtype=float))
        repeat_i += 1

    return y_all, scores_all


def mean_roc(y_all, scores_all, mean_fpr):
    interp_tprs = []
    aucs = []
    sens_at_spec80_all = []
    spec_at_sens80_all = []

    for y_true, scores in zip(y_all, scores_all):
        fpr, tpr, _ = roc_curve(y_true, scores)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0

        interp_tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

        # Sensitivity at 80% specificity:
        # specificity = 0.80 -> FPR = 0.20
        sens_at_spec80 = np.interp(
            0.20,
            fpr,
            tpr,
        )

        # Specificity at 80% sensitivity:
        # sensitivity = 0.80 -> find corresponding FPR,
        # then specificity = 1 - FPR.
        fpr_at_sens80 = np.interp(
            0.80,
            tpr,
            fpr,
        )

        spec_at_sens80 = 1.0 - fpr_at_sens80

        sens_at_spec80_all.append(
            sens_at_spec80
        )

        spec_at_sens80_all.append(
            spec_at_sens80
        )

    interp_tprs = np.asarray(interp_tprs)
    aucs = np.asarray(aucs)
    sens_at_spec80_all = np.asarray(
        sens_at_spec80_all
    )
    spec_at_sens80_all = np.asarray(
        spec_at_sens80_all
    )

    mean_tpr = np.mean(
        interp_tprs,
        axis=0,
    )

    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0

    sd_tpr = np.std(
        interp_tprs,
        axis=0,
        ddof=0,
    )

    lower = np.maximum(
        mean_tpr - sd_tpr,
        0.0,
    )

    upper = np.minimum(
        mean_tpr + sd_tpr,
        1.0,
    )

    return {
        "mean_tpr": mean_tpr,
        "lower": lower,
        "upper": upper,
        "mean_auc": np.mean(aucs),
        "sd_auc": np.std(aucs, ddof=0),
        "mean_sens_at_spec80": np.mean(
            sens_at_spec80_all
        ),
        "sd_sens_at_spec80": np.std(
            sens_at_spec80_all,
            ddof=0,
        ),
        "mean_spec_at_sens80": np.mean(
            spec_at_sens80_all
        ),
        "sd_spec_at_sens80": np.std(
            spec_at_sens80_all,
            ddof=0,
        ),
        "n_repeats": len(aucs),
    }


def plot_task(path):
    data = np.load(
        path,
        allow_pickle=False,
    )

    task_name = str(
        data["task_name"].item()
    )

    mean_fpr = np.linspace(
        0.0,
        1.0,
        501,
    )

    safe_task = (
        task_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )

    table_rows = []

    # ========================================================
    # INTERPOLATED MEAN ROC PLOT
    # ========================================================

    fig_interp, ax_interp = plt.subplots(
        figsize=(9, 8)
    )

    # ========================================================
    # RAW ROC PLOT
    #
    # Every repeat is plotted directly from roc_curve without
    # interpolation. The corresponding mean interpolated curve
    # is overlaid as a stronger line for orientation.
    # ========================================================

    fig_raw, ax_raw = plt.subplots(
        figsize=(9, 8)
    )

    for model_name in MODEL_NAMES:
        y_all, scores_all = load_model_repeats(
            data,
            model_name,
        )

        if not y_all:
            print(
                f"Missing ROC data: "
                f"{task_name} - {model_name}"
            )
            continue

        roc_stats = mean_roc(
            y_all,
            scores_all,
            mean_fpr,
        )

        mean_tpr = roc_stats[
            "mean_tpr"
        ]

        lower = roc_stats[
            "lower"
        ]

        upper = roc_stats[
            "upper"
        ]

        mean_auc = roc_stats[
            "mean_auc"
        ]

        sd_auc = roc_stats[
            "sd_auc"
        ]

        # ----------------------------------------------------
        # Interpolated mean ROC.
        # ----------------------------------------------------

        line, = ax_interp.plot(
            mean_fpr,
            mean_tpr,
            label=(
                f"{MODEL_LABELS[model_name]} "
                f"({mean_auc:.2f} ± {sd_auc:.2f})"
            ),
        )

        ax_interp.fill_between(
            mean_fpr,
            lower,
            upper,
            color=line.get_color(),
            alpha=0.08,
            linewidth=0,
        )

        # ----------------------------------------------------
        # Raw repeat ROC curves.
        # ----------------------------------------------------

        raw_color = line.get_color()

        for y_true, scores in zip(
            y_all,
            scores_all,
        ):
            fpr_raw, tpr_raw, _ = roc_curve(
                y_true,
                scores,
            )

            ax_raw.step(
                fpr_raw,
                tpr_raw,
                where="post",
                color=raw_color,
                alpha=0.05,
                linewidth=0.8,
            )

        # Overlay interpolated mean on raw curves.
        ax_raw.plot(
            mean_fpr,
            mean_tpr,
            color=raw_color,
            linewidth=2.0,
            label=(
                f"{MODEL_LABELS[model_name]} "
                f"({mean_auc:.2f} ± {sd_auc:.2f})"
            ),
        )

        table_rows.append({
            "task": task_name,
            "model": MODEL_LABELS[
                model_name
            ],
            "model_full": model_name,
            "n_repeats": roc_stats[
                "n_repeats"
            ],
            "auroc_mean": mean_auc,
            "auroc_sd": sd_auc,
            "sensitivity_at_80_specificity_mean":
                roc_stats[
                    "mean_sens_at_spec80"
                ],
            "sensitivity_at_80_specificity_sd":
                roc_stats[
                    "sd_sens_at_spec80"
                ],
            "specificity_at_80_sensitivity_mean":
                roc_stats[
                    "mean_spec_at_sens80"
                ],
            "specificity_at_80_sensitivity_sd":
                roc_stats[
                    "sd_spec_at_sens80"
                ],
        })

    # ========================================================
    # FORMAT BOTH ROC PLOTS
    # ========================================================

    for ax in [
        ax_interp,
        ax_raw,
    ]:
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="black",
            linewidth=1.5,
            label="Chance",
        )

        ax.axvline(
            0.20,
            linestyle="--",
            color="grey",
            linewidth=1.5,
            label="80% specificity",
        )

        ax.axhline(
            0.80,
            linestyle="--",
            color="dimgray",
            linewidth=1.5,
            label="80% sensitivity",
        )

        ax.set_xlim(
            0,
            1,
        )

        ax.set_ylim(
            0,
            1,
        )

        ax.set_xlabel(
            "1 - Specificity"
        )

        ax.set_ylabel(
            "Sensitivity"
        )

        ax.legend(
            loc="lower right",
            framealpha=0.6,
            ncol=1,
        )

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

    ax_interp.set_title(
        f"{task_name} - interpolated mean ROC"
    )

    ax_raw.set_title(
        f"{task_name} - raw repeat ROC curves"
    )

    fig_interp.tight_layout()
    fig_raw.tight_layout()

    interp_png = (
        OUTPUT_DIR /
        f"{safe_task}_roc_interpolated.png"
    )

    interp_pdf = (
        OUTPUT_DIR /
        f"{safe_task}_roc_interpolated.pdf"
    )

    raw_png = (
        OUTPUT_DIR /
        f"{safe_task}_roc_raw.png"
    )

    raw_pdf = (
        OUTPUT_DIR /
        f"{safe_task}_roc_raw.pdf"
    )

    fig_interp.savefig(
        interp_png,
        dpi=300,
        bbox_inches="tight",
    )

    fig_interp.savefig(
        interp_pdf,
        bbox_inches="tight",
    )

    fig_raw.savefig(
        raw_png,
        dpi=300,
        bbox_inches="tight",
    )

    fig_raw.savefig(
        raw_pdf,
        bbox_inches="tight",
    )

    plt.show()

    plt.close(
        fig_interp
    )

    plt.close(
        fig_raw
    )

    print(
        f"Saved: {interp_png}"
    )

    print(
        f"Saved: {interp_pdf}"
    )

    print(
        f"Saved: {raw_png}"
    )

    print(
        f"Saved: {raw_pdf}"
    )

    # ========================================================
    # PER-TASK TABLE
    # ========================================================

    task_table = pd.DataFrame(
        table_rows
    )

    if not task_table.empty:
        task_csv = (
            OUTPUT_DIR /
            f"{safe_task}_operating_points.csv"
        )

        task_table.to_csv(
            task_csv,
            index=False,
        )

        print(
            f"Saved: {task_csv}"
        )

        display_table = (
            task_table[
                [
                    "model",
                    "auroc_mean",
                    "auroc_sd",
                    "sensitivity_at_80_specificity_mean",
                    "sensitivity_at_80_specificity_sd",
                    "specificity_at_80_sensitivity_mean",
                    "specificity_at_80_sensitivity_sd",
                ]
            ]
            .copy()
        )

        display_table[
            "AUROC"
        ] = (
            display_table[
                "auroc_mean"
            ].map(
                lambda x:
                    f"{x:.3f}"
            )
            + " ± "
            + display_table[
                "auroc_sd"
            ].map(
                lambda x:
                    f"{x:.3f}"
            )
        )

        display_table[
            "Sensitivity @ 80% specificity"
        ] = (
            display_table[
                "sensitivity_at_80_specificity_mean"
            ].map(
                lambda x:
                    f"{x:.3f}"
            )
            + " ± "
            + display_table[
                "sensitivity_at_80_specificity_sd"
            ].map(
                lambda x:
                    f"{x:.3f}"
            )
        )

        display_table[
            "Specificity @ 80% sensitivity"
        ] = (
            display_table[
                "specificity_at_80_sensitivity_mean"
            ].map(
                lambda x:
                    f"{x:.3f}"
            )
            + " ± "
            + display_table[
                "specificity_at_80_sensitivity_sd"
            ].map(
                lambda x:
                    f"{x:.3f}"
            )
        )

        print(
            "\n"
            f"{task_name} operating-point table:"
        )

        print(
            display_table[
                [
                    "model",
                    "AUROC",
                    "Sensitivity @ 80% specificity",
                    "Specificity @ 80% sensitivity",
                ]
            ].to_string(
                index=False
            )
        )

    return table_rows


def main():
    all_rows = []

    for filename in TASK_FILES:
        path = (
            INPUT_DIR /
            filename
        )

        if not path.exists():
            print(
                f"Not found: {path}"
            )
            continue

        all_rows.extend(
            plot_task(path)
        )

    if not all_rows:
        return

    results = pd.DataFrame(
        all_rows
    )

    # ========================================================
    # FULL RAW NUMERIC SUMMARY
    # ========================================================

    combined_path = (
        OUTPUT_DIR /
        "all_tasks_AUROC_sensitivity_specificity_table.csv"
    )

    results.to_csv(
        combined_path,
        index=False,
    )

    # ========================================================
    # FULL FORMATTED SUMMARY
    # ========================================================

    formatted = results[
        [
            "task",
            "model",
            "auroc_mean",
            "auroc_sd",
            "sensitivity_at_80_specificity_mean",
            "sensitivity_at_80_specificity_sd",
            "specificity_at_80_sensitivity_mean",
            "specificity_at_80_sensitivity_sd",
        ]
    ].copy()

    formatted["AUROC"] = (
        formatted["auroc_mean"]
        .map(
            lambda x:
                f"{x:.3f}"
        )
        + " ± "
        + formatted["auroc_sd"]
        .map(
            lambda x:
                f"{x:.3f}"
        )
    )

    formatted[
        "Sensitivity @ 80% specificity"
    ] = (
        formatted[
            "sensitivity_at_80_specificity_mean"
        ]
        .map(
            lambda x:
                f"{x:.3f}"
        )
        + " ± "
        + formatted[
            "sensitivity_at_80_specificity_sd"
        ]
        .map(
            lambda x:
                f"{x:.3f}"
        )
    )

    formatted[
        "Specificity @ 80% sensitivity"
    ] = (
        formatted[
            "specificity_at_80_sensitivity_mean"
        ]
        .map(
            lambda x:
                f"{x:.3f}"
        )
        + " ± "
        + formatted[
            "specificity_at_80_sensitivity_sd"
        ]
        .map(
            lambda x:
                f"{x:.3f}"
        )
    )

    formatted = formatted[
        [
            "task",
            "model",
            "AUROC",
            "Sensitivity @ 80% specificity",
            "Specificity @ 80% sensitivity",
        ]
    ]

    formatted_path = (
        OUTPUT_DIR /
        "all_tasks_AUROC_sensitivity_specificity_formatted.csv"
    )

    formatted.to_csv(
        formatted_path,
        index=False,
    )

    # ========================================================
    # BEST MODEL FOR EACH TASK, SELECTED ONLY BY MEAN AUROC
    # ========================================================

    best_indices = (
        results.groupby(
            "task"
        )["auroc_mean"]
        .idxmax()
    )

    best_results = (
        results.loc[
            best_indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    best_raw_path = (
        OUTPUT_DIR /
        "best_model_each_task_by_AUROC_raw.csv"
    )

    best_results.to_csv(
        best_raw_path,
        index=False,
    )

    best_formatted = best_results[
        [
            "task",
            "model",
            "auroc_mean",
            "auroc_sd",
            "sensitivity_at_80_specificity_mean",
            "sensitivity_at_80_specificity_sd",
            "specificity_at_80_sensitivity_mean",
            "specificity_at_80_sensitivity_sd",
        ]
    ].copy()

    best_formatted["AUROC"] = (
        best_formatted[
            "auroc_mean"
        ].map(
            lambda x:
                f"{x:.3f}"
        )
        + " ± "
        + best_formatted[
            "auroc_sd"
        ].map(
            lambda x:
                f"{x:.3f}"
        )
    )

    best_formatted[
        "Sensitivity @ 80% specificity"
    ] = (
        best_formatted[
            "sensitivity_at_80_specificity_mean"
        ].map(
            lambda x:
                f"{x:.3f}"
        )
        + " ± "
        + best_formatted[
            "sensitivity_at_80_specificity_sd"
        ].map(
            lambda x:
                f"{x:.3f}"
        )
    )

    best_formatted[
        "Specificity @ 80% sensitivity"
    ] = (
        best_formatted[
            "specificity_at_80_sensitivity_mean"
        ].map(
            lambda x:
                f"{x:.3f}"
        )
        + " ± "
        + best_formatted[
            "specificity_at_80_sensitivity_sd"
        ].map(
            lambda x:
                f"{x:.3f}"
        )
    )

    best_formatted = best_formatted[
        [
            "task",
            "model",
            "AUROC",
            "Sensitivity @ 80% specificity",
            "Specificity @ 80% sensitivity",
        ]
    ]

    best_formatted_path = (
        OUTPUT_DIR /
        "best_model_each_task_by_AUROC_summary.csv"
    )

    best_formatted.to_csv(
        best_formatted_path,
        index=False,
    )

    # ========================================================
    # PRINT TABLES
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "ALL TASKS: AUROC AND FIXED OPERATING-POINT METRICS"
    )

    print(
        "=" * 120
    )

    print(
        formatted.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "BEST MODEL FOR EACH TASK BY MEAN AUROC"
    )

    print(
        "=" * 120
    )

    print(
        best_formatted.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {combined_path}"
    )

    print(
        f"Saved: {formatted_path}"
    )

    print(
        f"Saved: {best_raw_path}"
    )

    print(
        f"Saved: {best_formatted_path}"
    )


if __name__ == "__main__":
    main()

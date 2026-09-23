# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

from matplotlib import colors


# Apply the plotting style used for Grad-CAM figures.
def apply_gradcam_plot_style():
    plt.rcdefaults()

    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.linewidth": 1.4,
        "savefig.dpi": 300,
    })


N_REPEATS = 50
OUTER_TEST_SIZE = 0.30
INNER_VALIDATION_SIZE = 0.20

CNN_KERNEL_SIZE = 3
CNN_DROPOUT = 0.0

N_SAMPLES_PER_CLASS = 10

# from the task-level mean THz waveform.
IMPULSE_SEARCH_START = 150
IMPULSE_SEARCH_END = 260

PEAK_HALF_WIDTH = 10
TROUGH_HALF_WIDTH = 10

MORPHOLOGY_SCALES = np.array(
    [0.90, 0.95, 1.00, 1.05, 1.10],
    dtype=float,
)

EDGE_BLEND_POINTS = 4

# Save full [229, 400] example arrays as compressed NPZ files.
SAVE_EXAMPLE_ARRAYS = True


MODEL_DIR_CANDIDATES = [
    # Path("outputs/final_gradcam_signed") / "saved_models",
    Path("outputs/gradcam_models"),
]


OUTPUT_DIR = Path(
    "outputs/perturbation_peak_trough"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TASKS = [
    (
        "Dry vs Healthy - no moisturizer",
        "outputs/Xy/Xy_dry-skin-no-moist.npz",
        "paired_regions",
        "Healthy",
        "Dry",
    ),

    (
        "Dry vs Healthy - 10 min moisturizer",
        "outputs/Xy/Xy_dry-skin-10-moist.npz",
        "paired_regions",
        "Healthy",
        "Dry",
    ),

    (
        "Dry vs Healthy - 20 min moisturizer",
        "outputs/Xy/Xy_dry-skin-20-moist.npz",
        "paired_regions",
        "Healthy",
        "Dry",
    ),

    (
        "Eczema vs Psoriasis",
        "outputs/Xy/Xy_dry-skin-type.npz",
        "single_label",
        "Eczema",
        "Psoriasis",
    ),

    (
        "Skin Cancer vs Healthy",
        "outputs/Xy/Xy_skin-cancer.npz",
        "paired_regions",
        "Healthy",
        "Skin cancer",
    ),
]


TASKS_TO_RUN = [
    t[0]
    for t in TASKS
]


# Create patient IDs for patient-independent splitting.
def make_patient_ids(
    y,
    structure,
):
    if structure == "single_label":
        unique_patients = np.arange(
            len(y)
        )

        patient_ids = (
            unique_patients.copy()
        )

    elif structure == "paired_regions":
        if len(y) % 2:
            raise ValueError(
                "paired_regions requires "
                "an even number of samples"
            )

        n_patients = (
            len(y) // 2
        )

        unique_patients = np.arange(
            n_patients
        )

        patient_ids = np.concatenate(
            [
                unique_patients,
                unique_patients,
            ]
        )

    else:
        raise ValueError(
            "Unknown structure"
        )

    return (
        unique_patients,
        patient_ids,
    )


# Create train, validation and test patient splits.
def split_patients(
    y,
    unique_patients,
    structure,
    repeat_i,
):
    if structure == "single_label":
        train_pool, test_patients = (
            train_test_split(
                unique_patients,
                test_size=OUTER_TEST_SIZE,
                random_state=repeat_i,
                stratify=y,
            )
        )

        (
            train_patients,
            val_patients,
        ) = train_test_split(
            train_pool,
            test_size=INNER_VALIDATION_SIZE,
            random_state=repeat_i,
            stratify=y[train_pool],
        )

    else:
        train_pool, test_patients = (
            train_test_split(
                unique_patients,
                test_size=OUTER_TEST_SIZE,
                random_state=repeat_i,
            )
        )

        (
            train_patients,
            val_patients,
        ) = train_test_split(
            train_pool,
            test_size=INNER_VALIDATION_SIZE,
            random_state=repeat_i,
        )

    return (
        train_patients,
        val_patients,
        test_patients,
    )


# Two-block 1D CNN used for the final classifier.
class TwoBlockCNN1D(
    nn.Module
):
    # Build the two-block CNN.
    def __init__(
        self,
        input_channels=229,
        input_length=400,
    ):
        super().__init__()

        p = (
            CNN_KERNEL_SIZE // 2
        )

        self.features = nn.Sequential(
            nn.Conv1d(
                input_channels,
                64,
                CNN_KERNEL_SIZE,
                padding=p,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                64,
                128,
                CNN_KERNEL_SIZE,
                padding=p,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        with torch.no_grad():
            dummy = torch.zeros(
                1,
                input_channels,
                input_length,
            )

            flattened_features = (
                self.features(
                    dummy
                ).numel()
            )

        self.dropout = (
            nn.Dropout(
                CNN_DROPOUT
            )
            if CNN_DROPOUT > 0
            else nn.Identity()
        )

        self.fc = nn.Linear(
            flattened_features,
            2,
        )

    # Run the forward pass.
    def forward(
        self,
        x,
    ):
        x = self.features(
            x
        )

        x = torch.flatten(
            x,
            start_dim=1,
        )

        x = self.dropout(
            x
        )

        return self.fc(
            x
        )


# Reshape flattened measurements to [N, 229, 400] for the CNN.
def prepare_cnn_X(
    X,
    y,
):
    return torch.tensor(
        X.reshape(
            len(y),
            400,
            229,
            order="F",
        ),
        dtype=torch.float32,
    ).permute(
        0,
        2,
        1,
    )


# Load a saved CNN checkpoint.
def load_model_checkpoint(
    model_path,
    device,
):
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
        input_channels=int(
            checkpoint.get(
                "input_channels",
                229,
            )
        ),
        input_length=int(
            checkpoint.get(
                "input_length",
                400,
            )
        ),
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return (
        model,
        checkpoint,
    )


# Return the target-class logit minus the other-class logit.
def target_contrast(
    logits,
    target_class,
):
    target_class = int(
        target_class
    )

    other_class = (
        1 - target_class
    )

    return (
        logits[
            :,
            target_class,
        ]
        -
        logits[
            :,
            other_class,
        ]
    )


# Return target logit contrast and softmax probability.
def get_target_score_and_probability(
    model,
    x,
    target_class,
):
    with torch.no_grad():
        logits = model(
            x
        )

        score = target_contrast(
            logits,
            target_class,
        )

        probability = (
            torch.softmax(
                logits,
                dim=1,
            )[
                :,
                int(target_class),
            ]
        )

    return (
        score.detach()
        .cpu()
        .numpy(),

        probability.detach()
        .cpu()
        .numpy(),
    )


# Locate the main positive peak and following trough.
def find_peak_and_trough(
    mean_signal,
):
    lo = (
        IMPULSE_SEARCH_START
    )

    hi = min(
        IMPULSE_SEARCH_END,
        len(mean_signal),
    )

    segment = (
        mean_signal[
            lo:hi
        ]
    )

    peak_idx = (
        lo
        +
        int(
            np.argmax(
                segment
            )
        )
    )

    trough_start = min(
        peak_idx + 1,
        hi - 1,
    )

    trough_segment = (
        mean_signal[
            trough_start:hi
        ]
    )

    trough_idx = (
        trough_start
        +
        int(
            np.argmin(
                trough_segment
            )
        )
    )

    return (
        peak_idx,
        trough_idx,
    )


# Create a bounded window around a temporal index.
def clipped_window(
    center,
    half_width,
    n_points,
):
    left = max(
        0,
        int(
            center
            - half_width
        ),
    )

    right = min(
        n_points,
        int(
            center
            + half_width
            + 1
        ),
    )

    return (
        left,
        right,
    )


# Select samples belonging to one class.
def select_class_indices(
    y,
    class_i,
):
    indices = np.where(
        y == class_i
    )[0]

    if (
        N_SAMPLES_PER_CLASS
        is not None
    ):
        indices = indices[
            :N_SAMPLES_PER_CLASS
        ]

    return indices


# Construct a straight baseline across a local waveform segment.
def linear_local_baseline(
    segment,
):
    n = (
        segment.shape[-1]
    )

    alpha = torch.linspace(
        0.0,
        1.0,
        n,
        device=segment.device,
        dtype=segment.dtype,
    ).view(
        1,
        1,
        -1,
    )

    y0 = (
        segment[
            ...,
            :1,
        ]
    )

    y1 = (
        segment[
            ...,
            -1:,
        ]
    )

    return (
        y0
        +
        alpha
        * (
            y1 - y0
        )
    )


# Create smooth edge weights for local waveform edits.
def edge_blend(
    n,
    edge_points,
    device,
    dtype,
):
    blend = torch.ones(
        n,
        device=device,
        dtype=dtype,
    )

    m = min(
        edge_points,
        n // 2,
    )

    if m > 0:
        ramp = 0.5 * (
            1.0
            -
            torch.cos(
                torch.linspace(
                    0.0,
                    np.pi,
                    m + 2,
                    device=device,
                    dtype=dtype,
                )
            )
        )[
            1:-1
        ]

        blend[
            :m
        ] = ramp

        blend[
            -m:
        ] = torch.flip(
            ramp,
            dims=[0],
        )

    return blend.view(
        1,
        1,
        -1,
    )


# Scale the existing local waveform excursion around its baseline.
def scale_existing_morphology(
    x,
    left,
    right,
    scale,
):
    x_new = x.clone()

    segment = x[
        ...,
        left:right,
    ]

    baseline = (
        linear_local_baseline(
            segment
        )
    )

    target = (
        baseline
        +
        float(scale)
        * (
            segment
            - baseline
        )
    )

    blend = edge_blend(
        segment.shape[-1],
        EDGE_BLEND_POINTS,
        segment.device,
        segment.dtype,
    )

    x_new[
        ...,
        left:right,
    ] = (
        segment
        +
        blend
        * (
            target
            - segment
        )
    )

    return x_new


# Measure model response across peak and trough perturbations.
def peak_trough_2d_response(
    model,
    x,
    target_class,
    peak_left,
    peak_right,
    trough_left,
    trough_right,
):
    (
        original_score,
        original_probability,
    ) = get_target_score_and_probability(
        model,
        x,
        target_class,
    )

    n_scales = len(
        MORPHOLOGY_SCALES
    )

    delta_score = np.zeros(
        (
            n_scales,
            n_scales,
        ),
        dtype=float,
    )

    delta_probability = (
        np.zeros_like(
            delta_score
        )
    )

    for (
        i,
        peak_scale,
    ) in enumerate(
        MORPHOLOGY_SCALES
    ):
        for (
            j,
            trough_scale,
        ) in enumerate(
            MORPHOLOGY_SCALES
        ):
            x_mod = (
                scale_existing_morphology(
                    x,
                    peak_left,
                    peak_right,
                    peak_scale,
                )
            )

            x_mod = (
                scale_existing_morphology(
                    x_mod,
                    trough_left,
                    trough_right,
                    trough_scale,
                )
            )

            (
                score,
                probability,
            ) = (
                get_target_score_and_probability(
                    model,
                    x_mod,
                    target_class,
                )
            )

            delta_score[
                i,
                j,
            ] = np.mean(
                score
                - original_score
            )

            delta_probability[
                i,
                j,
            ] = np.mean(
                probability
                - original_probability
            )

    return (
        delta_score,
        delta_probability,
    )


# Plot and save the peak-versus-trough response heatmap.
def save_heatmap(
    mean_matrix,
    task_name,
    class_name,
    output_path,
):

    apply_gradcam_plot_style()

    pct = (
        MORPHOLOGY_SCALES
        - 1.0
    ) * 100.0


    base_cmap = plt.get_cmap("viridis")

    truncated_cmap = colors.LinearSegmentedColormap.from_list(
        "PiYG_truncated",
        base_cmap(
            np.linspace(
                0.1,
                0.9,
                256,
            )
        ),
    )


    vmax = np.max(
        np.abs(
            mean_matrix
        )
    )

    norm = colors.TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0.0,
        vmax=vmax,
    )

    # SAME SIZE AS GRAD-CAM
    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    im = ax.imshow(
        mean_matrix,
        origin="lower",
        aspect="auto",
        cmap=truncated_cmap,
        norm=norm,
    )


    ax.set_xticks(
        np.arange(
            len(pct)
        )
    )

    ax.set_yticks(
        np.arange(
            len(pct)
        )
    )

    ax.set_xticklabels(
        [
            f"{v:+.0f}%"
            for v in pct
        ],
        fontsize=16,
    )

    ax.set_yticklabels(
        [
            f"{v:+.0f}%"
            for v in pct
        ],
        fontsize=16,
    )

    ax.tick_params(
        axis="both",
        labelsize=16,
        width=1.4,
        length=6,
    )


    ax.set_xlabel(
        "Trough scaling",
        fontsize=18,
        labelpad=10,
    )

    ax.set_ylabel(
        "Peak scaling",
        fontsize=18,
        labelpad=10,
    )

    #     fontsize=18,
    #     pad=12,


    cbar = fig.colorbar(
        im,
        ax=ax,
        pad=0.04,
    )

    cbar.set_label(
        f"Change in {class_name} evidence (Δ logit)",
        fontsize=18,
        labelpad=12,
    )

    cbar.ax.tick_params(
        direction="in",
        labelsize=16,
        width=1.4,
        length=5,
    )


    for i in range(
        mean_matrix.shape[0]
    ):
        for j in range(
            mean_matrix.shape[1]
        ):
            ax.text(
                j,
                i,
                f"{mean_matrix[i, j]:+.3f}",
                ha="center",
                va="center",
                fontsize=16,
                color="black",
            )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plt.close(
        fig
    )


# Plot response to peak scaling with the trough unchanged.
def plot_peak_only_response(
    task_name,
    class0_name,
    class1_name,
    class0_grid,
    class1_grid,
    output_path,
):
    apply_gradcam_plot_style()

    original_idx = int(
        np.argmin(
            np.abs(
                MORPHOLOGY_SCALES
                - 1.0
            )
        )
    )

    x = (
        MORPHOLOGY_SCALES
        - 1.0
    ) * 100.0

    peak0 = class0_grid[
        :,
        :,
        original_idx,
    ]

    peak1 = class1_grid[
        :,
        :,
        original_idx,
    ]

    mean0 = peak0.mean(axis=0)
    std0 = peak0.std(axis=0)

    mean1 = peak1.mean(axis=0)
    std1 = peak1.std(axis=0)

    fig, ax = plt.subplots(
        figsize=(6, 8)
    )

    ax.fill_between(
        x,
        mean0 - std0,
        mean0 + std0,
        color="blue",
        alpha=0.5,
        linewidth=0,
    )

    ax.plot(
        x,
        mean0,
        color="blue",
        linewidth=2.5,
        marker="o",
        label=class0_name,
    )

    ax.fill_between(
        x,
        mean1 - std1,
        mean1 + std1,
        color="red",
        alpha=0.5,
        linewidth=0,
    )

    ax.plot(
        x,
        mean1,
        color="red",
        linewidth=2.5,
        marker="o",
        label=class1_name,
    )

    ax.axhline(
        0,
        color="black",
        linewidth=1,
    )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Positive-peak morphology scaling (%)"
    )

    ax.set_ylabel(
        "Change in target-vs-other logit"
    )

    ax.set_title(
        f"{task_name}\n"
        "Peak response (trough fixed)"
    )

    ax.legend(
        loc="lower left",
        framealpha=0.5,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# Plot response to trough scaling with the peak unchanged.
def plot_trough_only_response(
    task_name,
    class0_name,
    class1_name,
    class0_grid,
    class1_grid,
    output_path,
):
    apply_gradcam_plot_style()

    original_idx = int(
        np.argmin(
            np.abs(
                MORPHOLOGY_SCALES
                - 1.0
            )
        )
    )

    x = (
        MORPHOLOGY_SCALES
        - 1.0
    ) * 100.0

    trough0 = class0_grid[
        :,
        original_idx,
        :,
    ]

    trough1 = class1_grid[
        :,
        original_idx,
        :,
    ]

    mean0 = trough0.mean(axis=0)
    std0 = trough0.std(axis=0)

    mean1 = trough1.mean(axis=0)
    std1 = trough1.std(axis=0)

    fig, ax = plt.subplots(
        figsize=(6, 8)
    )

    ax.fill_between(
        x,
        mean0 - std0,
        mean0 + std0,
        color="blue",
        alpha=0.5,
        linewidth=0,
    )

    ax.plot(
        x,
        mean0,
        color="blue",
        linewidth=2.5,
        marker="o",
        label=class0_name,
    )

    ax.fill_between(
        x,
        mean1 - std1,
        mean1 + std1,
        color="red",
        alpha=0.5,
        linewidth=0,
    )

    ax.plot(
        x,
        mean1,
        color="red",
        linewidth=2.5,
        marker="o",
        label=class1_name,
    )

    ax.axhline(
        0,
        color="black",
        linewidth=1,
    )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Following-trough morphology scaling (%)"
    )

    ax.set_ylabel(
        "Change in target-vs-other logit"
    )

    ax.set_title(
        f"{task_name}\n"
        "Trough response (peak fixed)"
    )

    ax.legend(
        loc="lower left",
        framealpha=0.5,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# Average the 229 input pulses into one waveform for plotting.
def mean_waveform(
    x,
):
    return (
        x.mean(
            dim=1
        )
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )


# Save representative perturbed arrays.
def save_example_npz(
    output_path,
    **arrays,
):
    if not SAVE_EXAMPLE_ARRAYS:
        return

    cleaned = {}

    for (
        key,
        value,
    ) in arrays.items():
        if torch.is_tensor(
            value
        ):
            cleaned[
                key
            ] = (
                value.squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
        else:
            cleaned[
                key
            ] = value

    np.savez_compressed(
        output_path,
        **cleaned,
    )


# Plot representative peak and trough perturbations.
def plot_example_peak_trough(
    x,
    peak_left,
    peak_right,
    trough_left,
    trough_right,
    task_name,
    class_name,
    output_path,
    npz_path,
):
    x_both_low = (
        scale_existing_morphology(
            x,
            peak_left,
            peak_right,
            0.90,
        )
    )

    x_both_low = (
        scale_existing_morphology(
            x_both_low,
            trough_left,
            trough_right,
            0.90,
        )
    )

    x_both_high = (
        scale_existing_morphology(
            x,
            peak_left,
            peak_right,
            1.10,
        )
    )

    x_both_high = (
        scale_existing_morphology(
            x_both_high,
            trough_left,
            trough_right,
            1.10,
        )
    )

    x_peak_high_trough_low = (
        scale_existing_morphology(
            x,
            peak_left,
            peak_right,
            1.10,
        )
    )

    x_peak_high_trough_low = (
        scale_existing_morphology(
            x_peak_high_trough_low,
            trough_left,
            trough_right,
            0.90,
        )
    )

    x_peak_low_trough_high = (
        scale_existing_morphology(
            x,
            peak_left,
            peak_right,
            0.90,
        )
    )

    x_peak_low_trough_high = (
        scale_existing_morphology(
            x_peak_low_trough_high,
            trough_left,
            trough_right,
            1.10,
        )
    )

    original = mean_waveform(
        x
    )

    both_low = mean_waveform(
        x_both_low
    )

    both_high = mean_waveform(
        x_both_high
    )

    peak_high_trough_low = (
        mean_waveform(
            x_peak_high_trough_low
        )
    )

    peak_low_trough_high = (
        mean_waveform(
            x_peak_low_trough_high
        )
    )

    left = max(
        0,
        peak_left - 20,
    )

    right = min(
        len(original),
        trough_right + 20,
    )

    xx = np.arange(
        left,
        right,
    )

    apply_gradcam_plot_style()

    fig, ax = plt.subplots(
        figsize=(6, 8)
    )

    ax.plot(
        xx,
        original[
            left:right
        ],
        linewidth=2.4,
        label="Original",
    )

    ax.plot(
        xx,
        both_low[
            left:right
        ],
        linestyle="--",
        linewidth=1.8,
        label="Peak -10%, trough -10%",
    )

    ax.plot(
        xx,
        both_high[
            left:right
        ],
        linestyle="--",
        linewidth=1.8,
        label="Peak +10%, trough +10%",
    )

    ax.plot(
        xx,
        peak_high_trough_low[
            left:right
        ],
        linestyle=":",
        linewidth=1.8,
        label="Peak +10%, trough -10%",
    )

    ax.plot(
        xx,
        peak_low_trough_high[
            left:right
        ],
        linestyle=":",
        linewidth=1.8,
        label="Peak -10%, trough +10%",
    )

    ax.axvspan(
        peak_left,
        peak_right - 1,
        alpha=0.08,
    )

    ax.axvspan(
        trough_left,
        trough_right - 1,
        alpha=0.08,
    )

    ax.set_xlabel(
        "Data points"
    )

    ax.set_ylabel(
        "THz signal"
    )

    ax.set_title(
        f"{task_name} - "
        f"{class_name}\n"
        "Example peak × trough amplitude perturbations"
    )

    ax.legend(
        loc="lower left",
        framealpha=0.5,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    save_example_npz(
        npz_path,
        original=x,
        both_low=x_both_low,
        both_high=x_both_high,
        peak_high_trough_low=(
            x_peak_high_trough_low
        ),
        peak_low_trough_high=(
            x_peak_low_trough_high
        ),
    )


# Find the directory containing saved CNN models.
def find_model_root():
    for root in (
        MODEL_DIR_CANDIDATES
    ):
        if root.exists():
            return root

    raise FileNotFoundError(
        "Could not find saved models in:\n"
        +
        "\n".join(
            str(p)
            for p in (
                MODEL_DIR_CANDIDATES
            )
        )
    )


# Run the full analysis for one classification task.
def run_task(
    task_name,
    data_file,
    structure,
    class0_name,
    class1_name,
    model_root,
):
    print(
        "\n"
        + "=" * 90
    )

    print(
        task_name
    )

    print(
        "=" * 90
    )

    # LOAD DATA

    data = np.load(
        data_file
    )

    X = data[
        "X"
    ]

    y = data[
        "y"
    ].astype(
        int
    )

    (
        unique_patients,
        patient_ids,
    ) = make_patient_ids(
        y,
        structure,
    )

    X_cnn = prepare_cnn_X(
        X,
        y,
    )


    mean_signal = (
        X_cnn.mean(
            dim=1
        )
        .mean(
            dim=0
        )
        .cpu()
        .numpy()
    )

    (
        peak_idx,
        trough_idx,
    ) = find_peak_and_trough(
        mean_signal
    )

    n_points = (
        X_cnn.shape[-1]
    )

    (
        peak_left,
        peak_right,
    ) = clipped_window(
        peak_idx,
        PEAK_HALF_WIDTH,
        n_points,
    )

    (
        trough_left,
        trough_right,
    ) = clipped_window(
        trough_idx,
        TROUGH_HALF_WIDTH,
        n_points,
    )

    stem = (
        task_name.lower()
        .replace(
            " ",
            "_",
        )
        .replace(
            "/",
            "_",
        )
        .replace(
            "-",
            "_",
        )
    )

    task_model_dir = (
        model_root
        / stem
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Positive peak index: "
        f"{peak_idx}"
    )

    print(
        f"Following trough index: "
        f"{trough_idx}"
    )

    print(
        f"Peak window: "
        f"[{peak_left}, "
        f"{peak_right})"
    )

    print(
        f"Trough window: "
        f"[{trough_left}, "
        f"{trough_right})"
    )


    repeat_test_data = []

    for repeat_i in range(
        N_REPEATS
    ):
        model_path = (
            task_model_dir
            /
            f"repeat_{repeat_i:02d}.pt"
        )

        if not model_path.exists():
            continue

        (
            _,
            _,
            test_patients,
        ) = split_patients(
            y,
            unique_patients,
            structure,
            repeat_i,
        )

        test_mask = np.isin(
            patient_ids,
            test_patients,
        )

        X_test = (
            X_cnn[
                test_mask
            ]
        )

        y_test = (
            y[
                test_mask
            ]
        )

        (
            _,
            checkpoint,
        ) = load_model_checkpoint(
            model_path,
            device,
        )

        saved_test = np.asarray(
            checkpoint.get(
                "test_patients",
                test_patients,
            )
        )

        if not np.array_equal(
            saved_test,
            np.asarray(
                test_patients
            ),
        ):
            raise RuntimeError(
                f"Saved test split "
                f"does not match "
                f"repeat {repeat_i}"
            )

        repeat_test_data.append(
            (
                repeat_i,
                X_test,
                y_test,
            )
        )

    if len(
        repeat_test_data
    ) == 0:
        raise RuntimeError(
            "No matching saved models "
            f"found in {task_model_dir}"
        )

    print(
        f"Models used: "
        f"{len(repeat_test_data)}"
    )


    grid_score = {
        0: [],
        1: [],
    }

    grid_probability = {
        0: [],
        1: [],
    }

    example_saved = {
        0: False,
        1: False,
    }


    for (
        repeat_i,
        X_test,
        y_test,
    ) in repeat_test_data:

        model_path = (
            task_model_dir
            /
            f"repeat_{repeat_i:02d}.pt"
        )

        (
            model,
            _,
        ) = load_model_checkpoint(
            model_path,
            device,
        )

        for target_class in [
            0,
            1,
        ]:
            indices = (
                select_class_indices(
                    y_test,
                    target_class,
                )
            )

            if len(indices) == 0:
                continue

            x_class = (
                X_test[
                    indices
                ]
                .to(
                    device
                )
            )

            (
                ds,
                dp,
            ) = peak_trough_2d_response(
                model,
                x_class,
                target_class,
                peak_left,
                peak_right,
                trough_left,
                trough_right,
            )

            grid_score[
                target_class
            ].append(
                ds
            )

            grid_probability[
                target_class
            ].append(
                dp
            )

            # Save one representative example per class.

            if not example_saved[
                target_class
            ]:
                class_name = (
                    class0_name
                    if target_class == 0
                    else class1_name
                )

                x_example = (
                    x_class[
                        0:1
                    ]
                )

                plot_example_peak_trough(
                    x_example,
                    peak_left,
                    peak_right,
                    trough_left,
                    trough_right,
                    task_name,
                    class_name,
                    OUTPUT_DIR
                    /
                    (
                        f"{stem}_"
                        f"class{target_class}_"
                        f"example_peak_trough.png"
                    ),
                    OUTPUT_DIR
                    /
                    (
                        f"{stem}_"
                        f"class{target_class}_"
                        f"example_peak_trough.npz"
                    ),
                )

                example_saved[
                    target_class
                ] = True

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    class0_grid = np.stack(
        grid_score[
            0
        ]
    )

    class1_grid = np.stack(
        grid_score[
            1
        ]
    )

    class0_prob = np.stack(
        grid_probability[
            0
        ]
    )

    class1_prob = np.stack(
        grid_probability[
            1
        ]
    )


    save_heatmap(
        class0_grid.mean(
            axis=0
        ),
        task_name,
        class0_name,
        OUTPUT_DIR
        /
        (
            f"{stem}_"
            f"class0_"
            f"peak_trough_heatmap.png"
        ),
    )

    save_heatmap(
        class1_grid.mean(
            axis=0
        ),
        task_name,
        class1_name,
        OUTPUT_DIR
        /
        (
            f"{stem}_"
            f"class1_"
            f"peak_trough_heatmap.png"
        ),
    )


    plot_peak_only_response(
        task_name,
        class0_name,
        class1_name,
        class0_grid,
        class1_grid,
        OUTPUT_DIR
        /
        (
            f"{stem}_"
            f"peak_only_response_"
            f"mean_sd.png"
        ),
    )


    plot_trough_only_response(
        task_name,
        class0_name,
        class1_name,
        class0_grid,
        class1_grid,
        OUTPUT_DIR
        /
        (
            f"{stem}_"
            f"trough_only_response_"
            f"mean_sd.png"
        ),
    )

    # SAVE NUMERICAL RESULTS

    original_idx = int(
        np.argmin(
            np.abs(
                MORPHOLOGY_SCALES
                - 1.0
            )
        )
    )

    peak_only_class0 = (
        class0_grid[
            :,
            :,
            original_idx,
        ]
    )

    peak_only_class1 = (
        class1_grid[
            :,
            :,
            original_idx,
        ]
    )

    trough_only_class0 = (
        class0_grid[
            :,
            original_idx,
            :,
        ]
    )

    trough_only_class1 = (
        class1_grid[
            :,
            original_idx,
            :,
        ]
    )

    np.savez(
        OUTPUT_DIR
        /
        (
            f"{stem}_"
            f"peak_trough_amplitude_results.npz"
        ),

        morphology_scales=(
            MORPHOLOGY_SCALES
        ),

        peak_idx=np.asarray(
            peak_idx
        ),

        trough_idx=np.asarray(
            trough_idx
        ),

        peak_window=np.asarray(
            [
                peak_left,
                peak_right,
            ]
        ),

        trough_window=np.asarray(
            [
                trough_left,
                trough_right,
            ]
        ),

        mean_input_signal=(
            mean_signal
        ),

        grid_logit_change_class0=(
            class0_grid
        ),

        grid_logit_change_class1=(
            class1_grid
        ),

        grid_prob_change_class0=(
            class0_prob
        ),

        grid_prob_change_class1=(
            class1_prob
        ),

        peak_only_logit_change_class0=(
            peak_only_class0
        ),

        peak_only_logit_change_class1=(
            peak_only_class1
        ),

        trough_only_logit_change_class0=(
            trough_only_class0
        ),

        trough_only_logit_change_class1=(
            trough_only_class1
        ),
    )


    summary_path = (
        OUTPUT_DIR
        /
        f"{stem}_summary.txt"
    )

    pct = (
        MORPHOLOGY_SCALES
        - 1.0
    ) * 100.0

    with open(
        summary_path,
        "w",
    ) as f:
        f.write(
            f"{task_name}\n"
        )

        f.write(
            "=" * 90
            + "\n"
        )

        f.write(
            f"Class 0: "
            f"{class0_name}\n"
        )

        f.write(
            f"Class 1: "
            f"{class1_name}\n"
        )

        f.write(
            f"Used repeats: "
            f"{len(repeat_test_data)}\n"
        )

        f.write(
            f"Positive peak index: "
            f"{peak_idx}\n"
        )

        f.write(
            f"Following trough index: "
            f"{trough_idx}\n"
        )

        f.write(
            f"Peak window: "
            f"[{peak_left}, "
            f"{peak_right})\n"
        )

        f.write(
            f"Trough window: "
            f"[{trough_left}, "
            f"{trough_right})\n\n"
        )

        f.write(
            "ONLY peak/trough amplitude-morphology "
            "scaling was performed.\n"
        )

        f.write(
            "No temporal shift, separation, "
            "internal shift or broadening "
            "was performed.\n\n"
        )

        f.write(
            "SIGN CONVENTION\n"
        )

        f.write(
            "-" * 90
            + "\n"
        )

        f.write(
            "Score = target logit "
            "- other-class logit.\n"
        )

        f.write(
            "Positive change supports "
            "the target class.\n"
        )

        f.write(
            "Negative change opposes "
            "the target class.\n\n"
        )


        f.write(
            "PEAK-ONLY RESPONSE "
            "(trough fixed at original)\n"
        )

        f.write(
            "=" * 90
            + "\n"
        )

        for (
            class_name,
            arr,
        ) in [
            (
                class0_name,
                peak_only_class0,
            ),
            (
                class1_name,
                peak_only_class1,
            ),
        ]:
            f.write(
                f"\n{class_name}\n"
            )

            for i, p in enumerate(
                pct
            ):
                f.write(
                    f"{p:+.1f}%: "
                    f"{arr[:, i].mean():+.6f} "
                    f"+/- "
                    f"{arr[:, i].std():.6f}\n"
                )


        f.write(
            "\n\nTROUGH-ONLY RESPONSE "
            "(peak fixed at original)\n"
        )

        f.write(
            "=" * 90
            + "\n"
        )

        for (
            class_name,
            arr,
        ) in [
            (
                class0_name,
                trough_only_class0,
            ),
            (
                class1_name,
                trough_only_class1,
            ),
        ]:
            f.write(
                f"\n{class_name}\n"
            )

            for i, p in enumerate(
                pct
            ):
                f.write(
                    f"{p:+.1f}%: "
                    f"{arr[:, i].mean():+.6f} "
                    f"+/- "
                    f"{arr[:, i].std():.6f}\n"
                )


        f.write(
            "\n\nFULL 2-D PEAK x TROUGH MATRIX\n"
        )

        f.write(
            "=" * 90
            + "\n"
        )

        for (
            class_name,
            arr,
        ) in [
            (
                class0_name,
                class0_grid,
            ),
            (
                class1_name,
                class1_grid,
            ),
        ]:
            f.write(
                f"\n{class_name}\n"
            )

            mean_arr = arr.mean(
                axis=0
            )

            std_arr = arr.std(
                axis=0
            )

            for i, peak_pct in enumerate(
                pct
            ):
                for j, trough_pct in enumerate(
                    pct
                ):
                    f.write(
                        f"Peak "
                        f"{peak_pct:+.1f}%, "
                        f"Trough "
                        f"{trough_pct:+.1f}%: "
                        f"{mean_arr[i, j]:+.6f} "
                        f"+/- "
                        f"{std_arr[i, j]:.6f}\n"
                    )

    print(
        f"Saved results to: "
        f"{OUTPUT_DIR}"
    )


# Run the script.
def main():
    print(
        "Device:",
        (
            "CUDA"
            if torch.cuda.is_available()
            else "CPU"
        ),
    )

    model_root = (
        find_model_root()
    )

    print(
        "Using saved-model directory:",
        model_root,
    )

    print(
        "Output directory:",
        OUTPUT_DIR,
    )

    print(
        "\nRunning ONLY:"
    )

    print(
        "Peak x trough amplitude/"
        "morphology perturbation"
    )

    print(
        "Outputs:"
    )

    print(
        "  1. Class-specific 2-D heatmaps"
    )

    print(
        "  2. Peak-only line plot "
        "with mean +/- SD"
    )

    print(
        "  3. Trough-only line plot "
        "with mean +/- SD"
    )

    for task in TASKS:
        if (
            task[0]
            in TASKS_TO_RUN
        ):
            run_task(
                *task,
                model_root=model_root,
            )


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Visualise the peak/trough perturbation step-by-step.

Shows:
1. Original waveform region + local linear baseline
2. Excursion from baseline (residual)
3. Residual after scaling
4. Final smoothly blended perturbation

No CNN is used here.
This is only to visualise how the waveform perturbation is created.
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# SETTINGS
# =============================================================================

DATA_FILE = "outputs/Xy/Xy_dry-skin-no-moist.npz"

SAMPLE_INDEX = 0       # which measurement/patient sample
PULSE_INDEX = 40       # one of the 229 repeated pulses

REGION = "peak"        # "peak" or "trough"

SCALE = 1.10           # 1.10 = +10%, 0.90 = -10%

IMPULSE_SEARCH_START = 150
IMPULSE_SEARCH_END = 260

HALF_WIDTH = 10
EDGE_BLEND_POINTS = 4


# =============================================================================
# LOAD DATA
# =============================================================================

data = np.load(DATA_FILE)

X = data["X"]
y = data["y"]

# Same reshape as CNN code:
# [samples, 229 pulses, 400 temporal points]
X_cnn = X.reshape(
    len(y),
    400,
    229,
    order="F"
).transpose(0, 2, 1)

print("Data shape:", X_cnn.shape)


# =============================================================================
# FIND PEAK AND TROUGH
# =============================================================================

# Mean waveform only used to identify common peak/trough location
mean_signal = X_cnn.mean(axis=1).mean(axis=0)

search = mean_signal[
    IMPULSE_SEARCH_START:IMPULSE_SEARCH_END
]

peak_idx = (
    IMPULSE_SEARCH_START
    + np.argmax(search)
)

trough_idx = (
    peak_idx + 1
    + np.argmin(
        mean_signal[
            peak_idx + 1:
            IMPULSE_SEARCH_END
        ]
    )
)

print("Peak index:", peak_idx)
print("Trough index:", trough_idx)


# Choose which feature to perturb
if REGION.lower() == "peak":
    center = peak_idx

elif REGION.lower() == "trough":
    center = trough_idx

else:
    raise ValueError("REGION must be 'peak' or 'trough'")


left = max(
    0,
    center - HALF_WIDTH
)

right = min(
    400,
    center + HALF_WIDTH + 1
)


# =============================================================================
# SELECT ONE REAL THz PULSE
# =============================================================================

waveform = X_cnn[
    SAMPLE_INDEX,
    PULSE_INDEX,
].copy()

segment = waveform[
    left:right
].copy()

t = np.arange(
    left,
    right
)


# =============================================================================
# STEP 1 -- LOCAL LINEAR BASELINE
# =============================================================================

y0 = segment[0]
y1 = segment[-1]

alpha = np.linspace(
    0,
    1,
    len(segment)
)

baseline = (
    y0
    +
    alpha * (y1 - y0)
)


# =============================================================================
# STEP 2 -- EXCURSION FROM BASELINE
# =============================================================================

residual = (
    segment
    - baseline
)


# =============================================================================
# STEP 3 -- SCALE EXISTING MORPHOLOGY
# =============================================================================

scaled_residual = (
    SCALE
    * residual
)

target = (
    baseline
    + scaled_residual
)


# =============================================================================
# STEP 4 -- SMOOTH EDGE BLENDING
# =============================================================================

blend = np.ones(
    len(segment)
)

m = min(
    EDGE_BLEND_POINTS,
    len(segment) // 2
)

if m > 0:

    ramp = 0.5 * (
        1
        -
        np.cos(
            np.linspace(
                0,
                np.pi,
                m + 2
            )
        )
    )[1:-1]

    blend[:m] = ramp

    blend[-m:] = ramp[::-1]


# Final perturbed local segment
final_segment = (
    segment
    +
    blend
    * (
        target
        - segment
    )
)


# Put modified segment back into complete waveform
perturbed_waveform = waveform.copy()

perturbed_waveform[
    left:right
] = final_segment


# =============================================================================
# PLOT STEP 1
# =============================================================================

plt.figure(figsize=(8, 5))

plt.plot(
    t,
    segment,
    marker="o",
    linewidth=2,
    label="Original segment"
)

plt.plot(
    t,
    baseline,
    linestyle="--",
    linewidth=2,
    label="Local linear baseline"
)

plt.xlabel("Temporal sample")
plt.ylabel("THz signal")
plt.title("Step 1: Original waveform and local baseline")

plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# PLOT STEP 2
# =============================================================================

plt.figure(figsize=(8, 5))

plt.plot(
    t,
    residual,
    marker="o",
    linewidth=2,
    label="Original excursion"
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel("Temporal sample")
plt.ylabel("Signal − baseline")
plt.title("Step 2: Existing morphology relative to baseline")

plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# PLOT STEP 3
# =============================================================================

plt.figure(figsize=(8, 5))

plt.plot(
    t,
    residual,
    linewidth=2,
    label="Original residual"
)

plt.plot(
    t,
    scaled_residual,
    linewidth=2,
    linestyle="--",
    label=f"Scaled residual ({SCALE:.2f}×)"
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel("Temporal sample")
plt.ylabel("Excursion from baseline")
plt.title("Step 3: Scale the existing morphology")

plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# PLOT STEP 4
# =============================================================================

plt.figure(figsize=(8, 5))

plt.plot(
    t,
    segment,
    linewidth=2.5,
    label="Original"
)

plt.plot(
    t,
    target,
    linestyle="--",
    linewidth=2,
    label="Scaled before edge blending"
)

plt.plot(
    t,
    final_segment,
    linewidth=2.5,
    label="Final perturbation"
)

plt.plot(
    t,
    baseline,
    linestyle=":",
    linewidth=1.5,
    label="Baseline"
)

plt.xlabel("Temporal sample")
plt.ylabel("THz signal")
plt.title(
    f"Step 4: Final {REGION} perturbation "
    f"({(SCALE - 1) * 100:+.0f}%)"
)

plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# FINAL -- SHOW MODIFICATION IN COMPLETE WAVEFORM
# =============================================================================

plt.figure(figsize=(10, 5))

plt.plot(
    waveform,
    linewidth=2,
    label="Original waveform"
)

plt.plot(
    perturbed_waveform,
    linestyle="--",
    linewidth=2,
    label="Perturbed waveform"
)

plt.axvspan(
    left,
    right - 1,
    alpha=0.15,
    label="Perturbed region"
)

plt.xlabel("Temporal sample")
plt.ylabel("THz signal")
plt.title(
    f"Complete waveform: "
    f"{REGION} {(SCALE - 1) * 100:+.0f}%"
)

plt.legend()
plt.tight_layout()
plt.show()
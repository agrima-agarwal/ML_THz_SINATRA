# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from functions import THzData


OUTPUT_DIR = "outputs/Xy"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Store one patient's ROI and control THz measurements.
class patient_analysis:

    # Load ROI and control THz data for one patient.
    def __init__(self, folder, roi_folder="roi", control_folder="control"):
        try:
            self.patient_id = folder

            roi = THzData(
                folder + "/" + roi_folder,
                folder + "/reference.txt",
                folder + "/baseline.txt",
            )

            con = THzData(
                folder + "/" + control_folder,
                folder + "/reference.txt",
                folder + "/baseline.txt",
            )

            self.roi = roi
            self.con = con

        except Exception:
            print("no")


# Build and save one paired ROI/control dataset.
def generate_paired_dataset(
    raw_root,
    roi_folder,
    control_folder,
    black_list,
    output_name,
):

    D = []

    for j in tqdm(
        os.listdir(raw_root),
        desc=f"Loading {output_name}",
    ):
        if j not in black_list:
            Dd = patient_analysis(
                raw_root + "/" + j,
                roi_folder=roi_folder,
                control_folder=control_folder,
            )
            D.append(Dd)

    print(f"{output_name}: {len(D)} patients loaded")

    imp_all = []
    labels = []

    # Same exact ROI-then-control ordering as the original scripts.
    for region in ["roi", "con"]:

        imp_all1 = np.array([
            getattr(obj, region).impulses.T.flatten()
            for obj in D
        ])

        imp_all.extend(imp_all1)

        if region == "roi":
            lab = 1
        else:
            lab = 0

        label = np.repeat(lab, len(imp_all1))
        labels.extend(label)

    imp_all = np.array(imp_all)
    labels = np.array(labels, dtype=int)

    # Same filtering logic as original scripts.
    X = imp_all[np.where((labels == 0) | (labels == 1))]
    y = labels[(labels == 0) | (labels == 1)]

    save_path = os.path.join(OUTPUT_DIR, output_name)

    # Same saving format as before: keys X and y.
    np.savez(save_path, X=X, y=y)

    print(
        f"Saved {save_path}.npz | "
        f"X shape = {X.shape} | y shape = {y.shape}"
    )

    return X, y


# Build and save the eczema-versus-psoriasis dataset.
def generate_dry_skin_type():

    raw_root = "raw files dry skin"
    black_list = ["S003", "S006", "S036"]

    # Same Excel loading format as original.
    df = pd.read_excel(
        "dry_skin_type.xlsx",
        header=None,
    )

    D = []

    for j in tqdm(
        os.listdir(raw_root),
        desc="Loading Xy_dry-skin-type",
    ):
        if j not in black_list:

            Dd = patient_analysis(
                raw_root + "/" + j,
                roi_folder="roi",
                control_folder="control",
            )

            # Same label lookup as original.
            Dd.type = df.loc[
                df.iloc[:, 0] == j
            ].iloc[0, 1]

            D.append(Dd)

    print(f"Xy_dry-skin-type: {len(D)} patients loaded before e/p filtering")

    # Same exact feature construction as original:
    imp_all = np.array([
        obj.roi.impulses.T.flatten()
        for obj in D
    ])

    labels = np.array([
        obj.type
        for obj in D
    ])

    # Same e/p filtering as original.
    X = imp_all[
        np.where(
            (labels == "p") |
            (labels == "e")
        )
    ]

    y = labels[
        (labels == "p") |
        (labels == "e")
    ]

    # Same label coding as original.
    y = np.where(y == "p", 1, 0)

    save_path = os.path.join(
        OUTPUT_DIR,
        "Xy_dry-skin-type",
    )

    # Same saving format as before.
    np.savez(
        save_path,
        X=X,
        y=y,
    )

    print(
        f"Saved {save_path}.npz | "
        f"X shape = {X.shape} | y shape = {y.shape}"
    )

    return X, y


# Generate the dry-versus-healthy dataset before moisturiser.
def generate_dry_skin_no_moist():
    return generate_paired_dataset(
        raw_root="raw files dry skin",
        roi_folder="roi",
        control_folder="control",
        black_list=[
            "S003",
            "S006",
            "S036",
        ],
        output_name="Xy_dry-skin-no-moist",
    )


# Generate the dry-versus-healthy dataset 10 minutes after moisturiser.
def generate_dry_skin_10_moist():
    return generate_paired_dataset(
        raw_root="raw files dry skin",
        roi_folder="roi_10min",
        control_folder="control_10min",
        black_list=[
            "S003",
            "S006",
            "S014",
            "S016",
            "S027",
            "S028",
            "S036",
        ],
        output_name="Xy_dry-skin-10-moist",
    )


# Generate the dry-versus-healthy dataset 20 minutes after moisturiser.
def generate_dry_skin_20_moist():
    return generate_paired_dataset(
        raw_root="raw files dry skin",
        roi_folder="roi_20min",
        control_folder="control_20min",
        black_list=[
            "S003",
            "S008",
            "S012",
            "S016",
            "S027",
            "S036",
            "S063",
            "S073",
        ],
        output_name="Xy_dry-skin-20-moist",
    )


# Generate the skin-cancer-versus-healthy dataset.
def generate_skin_cancer():
    return generate_paired_dataset(
        raw_root="raw files skin cancer",
        roi_folder="roi",
        control_folder="control",
        black_list=[
            "S076",
        ],
        output_name="Xy_skin-cancer",
    )


# Run the script.
def main():

    print("\n" + "=" * 80)
    print("GENERATING ALL THz ML X/y DATASETS")
    print("=" * 80)

    print("\n[1/5] Dry vs Healthy - no moisturizer")
    generate_dry_skin_no_moist()

    print("\n[2/5] Dry vs Healthy - 10 min moisturizer")
    generate_dry_skin_10_moist()

    print("\n[3/5] Dry vs Healthy - 20 min moisturizer")
    generate_dry_skin_20_moist()

    print("\n[4/5] Eczema vs Psoriasis")
    generate_dry_skin_type()

    print("\n[5/5] Skin Cancer vs Healthy")
    generate_skin_cancer()

    print("\n" + "=" * 80)
    print("ALL FIVE DATASETS GENERATED")
    print("=" * 80)

    print("\nSaved files:")
    print("  outputs/Xy_dry-skin-no-moist.npz")
    print("  outputs/Xy_dry-skin-10-moist.npz")
    print("  outputs/Xy_dry-skin-20-moist.npz")
    print("  outputs/Xy_dry-skin-type.npz")
    print("  outputs/Xy_skin-cancer.npz")


if __name__ == "__main__":
    main()

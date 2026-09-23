# Machine Learning Analysis of In Vivo THz-TDS for Dermatological Classification

This repository contains the analysis code accompanying the manuscript on machine-learning classification and interpretation of in vivo terahertz time-domain spectroscopy (THz-TDS) measurements of human skin.



## Study overview

The study uses THz-TDS measurements from **70 patients**:

* 40 patients from the dry-skin cohort.
* 30 patients from the skin-cancer cohort.

For the paired classification datasets, measurements from the skin condition/region of interest are compared with healthy control skin from the same patient.

The five classification datasets analysed in the code are:

1. **Dry vs Healthy — no moisturizer**
2. **Dry vs Healthy — 10 min after moisturizer**
3. **Dry vs Healthy — 20 min after moisturizer**
4. **Eczema vs Psoriasis**
5. **Skin Cancer vs Healthy**

The machine-learning analyses use patient-independent train/validation/test splitting so that measurements from the same patient are not shared between evaluation partitions.



## Data representation

The processed THz measurement for one sample contains:

* **229 repeated THz impulses**
* **400 temporal points per impulse**

For the classical machine-learning methods, the data are stored as flattened feature vectors.

For the CNN analyses, each sample is reshaped to:

```text
[N samples, 229 impulses, 400 temporal points]
```

The generated `.npz` datasets use:

```text
X    processed THz measurements
y    binary class labels
```



## Repository scripts

|Script|Purpose|
|-|-|
|`generate_Xy_datasets.py`|Generates the five processed `Xy` datasets from the raw THz measurements.|
|`Train_all_models.py`|Trains and evaluates the classical ML models and the selected 1D CNN using repeated patient-independent splits.|
|`ROC_sens_spec_all_models.py`|Generates ROC plots and calculates AUROC, sensitivity at 80% specificity, and specificity at 80% sensitivity.|
|`CNN_ablation.py`|Compares alternative 1D CNN architectures across the five classification tasks.|
|`PCA_plots.py`|Generates mean THz response plots, PCA component plots, PC1 distributions, and associated PCA statistics.|
|`GradCAM_plots.py`|Trains/loads the selected CNN and generates class-specific 1D Grad-CAM analyses.|
|`CNN_Perturbation_peak_trough.py`|Perturbs the positive peak and following trough of the THz impulse and quantifies changes in CNN class evidence.|
|`Understanding_Perturbation_Procedure.py`|Visual demonstration of the waveform perturbation procedure without running a CNN.|
|`CNN_moist_robustness_and_Page_trend.py`|Trains the CNN at baseline and evaluates the same held-out dry-skin patients before and after moisturisation; includes Page trend analysis.|
|`PC1_moist_robustness_and_Page_trend.py`|Performs the corresponding baseline-trained PC1 hydration-robustness analysis and Page trend test.|

The analysis also requires the project-specific `functions.py` module containing the THz data-loading and preprocessing routines.



## Main ML models

The benchmarking script evaluates the following 11 approaches:

1. PC1
2. Linear Discriminant Analysis (LDA)
3. Logistic Regression
4. Logistic Regression + PCA
5. k-Nearest Neighbours (kNN)
6. kNN + PCA
7. Decision Tree
8. Random Forest
9. Single-Layer Perceptron (SLP)
10. Multi-Layer Perceptron (MLP)
11. 1D Convolutional Neural Network (CNN)

Hyperparameters for the classical models are selected from the parameter grids defined in `Train\\\_all\\\_models.py` using the validation data for each repeated split.



## CNN configuration

The selected CNN used by the final training and interpretation scripts contains two convolutional blocks:

```text
Input: 229 channels x 400 temporal points

Conv1D: 229 -> 64, kernel size 3
ReLU
MaxPool1D

Conv1D: 64 -> 128, kernel size 3
ReLU
MaxPool1D

Flatten
Fully connected layer -> 2 classes
```

Training settings used in the scripts include:

```text
Optimizer: Adam
Learning rate: 1e-3
Weight decay: 1e-4
Batch size: 8
Maximum epochs: 50
Early-stopping patience: 6
Loss: CrossEntropyLoss
```

In `CNN_ablation.py`, this selected two-convolution-block architecture is named `3_layer`.



## CNN ablation study

`CNN_ablation.py` compares:

```text
3_layer
4_layer
2_layer
no_max_pooling
global_avg_p
```

The analysis stores per-repeat results, architecture summaries, parameter counts, and paired AUROC comparisons against the reference `3_layer` architecture.

Outputs are written to:

```text
outputs/cnn_ablation/
```



## Model evaluation

The main classification experiments use repeated patient-independent splits.

The scripts report metrics including:

* AUROC
* accuracy
* sensitivity
* specificity
* ROC curves
* selected model hyperparameters
* per-repeat predictions/scores

`ROC_sens_spec_all_models.py` additionally calculates:

* **Sensitivity at 80% specificity**
* **Specificity at 80% sensitivity**

ROC curves are interpolated onto a common false-positive-rate grid for calculation and visualisation of mean ROC behaviour across repetitions. Raw repeat-level ROC curves are also saved.



## PCA analysis

`PCA_plots.py` performs PCA independently for each of the five datasets and generates:

* mean class THz responses
* first and second principal-component waveforms
* PC1 score distributions
* explained-variance statistics
* Welch's t-test of PC1 scores between classes

Outputs are written to:

```text
outputs/pca/
```



## Grad-CAM analysis

`GradCAM_plots.py` performs one-dimensional Grad-CAM on the selected CNN.

The Grad-CAM analysis is intended to identify temporal regions of the THz waveform that contribute strongly to the CNN classification decision. High Grad-CAM importance should therefore be interpreted as model reliance on a waveform region rather than simply as high THz signal amplitude.

The script generates class-specific Grad-CAM profiles and saves CNN checkpoints for repeated patient-independent splits.

Outputs are written under:

```text
outputs/gradcam/
```



## Peak/trough perturbation analysis

`CNN_Perturbation_peak_trough.py` complements Grad-CAM by directly modifying waveform morphology and measuring the resulting change in CNN evidence.

The script:

1. identifies the main positive peak and following trough of the THz impulse;
2. defines local windows around the peak and trough;
3. constructs a linear local baseline;
4. scales the existing waveform excursion relative to that baseline;
5. smoothly blends the modified region into the surrounding waveform;
6. evaluates the change in target-vs-other CNN logit.

The tested morphology scales are:

```text
90%, 95%, 100%, 105%, 110%
```

Positive and negative changes in target-class logit contrast indicate whether the perturbation shifts the CNN response towards or away from the target class.

`Understanding_Perturbation_Procedure.py` provides a step-by-step visualisation of the same perturbation operation without using the CNN.



## Moisturisation robustness

Two scripts examine whether dry-skin model scores change systematically after moisturiser application:

```text
CNN_moist_robustness_and_Page_trend.py
PC1_moist_robustness_and_Page_trend.py
```

For these experiments:

* only patients with measurements available at all three time points are retained;
* the same held-out patients are evaluated at all three conditions;
* the model/PCA transformation is trained using **no-moisturizer measurements only**;
* the trained transformation is frozen before evaluation at 0, 10, and 20 min;
* ordered changes are assessed using the **Page trend test**.

The tested sequence is:

```text
No moisturizer -> 10 min -> 20 min
```

CNN outputs are stored under:

```text
outputs/CNN_hydration_score/
```

PC1 outputs are stored under:

```text
outputs/PC1_hydration_score/
```



## Expected raw-data structure

The dataset-generation scripts assume a directory structure similar to:

```text
project/
|
|-- functions.py
|-- dry_skin_type.xlsx
|
|-- raw files dry skin/
|   |-- Sxxx/
|   |   |-- roi/
|   |   |-- control/
|   |   |-- roi_10min/
|   |   |-- control_10min/
|   |   |-- roi_20min/
|   |   |-- control_20min/
|   |   |-- reference.txt
|   |   `-- baseline.txt
|   `-- ...
|
|-- raw files skin cancer/
|   |-- Sxxx/
|   |   |-- roi/
|   |   |-- control/
|   |   |-- reference.txt
|   |   `-- baseline.txt
|   `-- ...
|
`-- outputs/
```

Some patients are excluded by dataset-specific quality-control blacklists defined directly in the scripts.



## Recommended execution order

A typical reproduction workflow is:

```text
1. generate_Xy_datasets.py
2. CNN_ablation.py
3. Train_all_models.py
4. ROC_sens_spec_all_models.py 
5. PCA_plots.py
6. GradCAM_plots.py
7. CNN_Perturbation_peak_trough.py
8. CNN_moist_robustness_and_Page_trend.py
9. PC1_moist_robustness_and_Page_trend.py
```

`Understanding_Perturbation_Procedure.py` can be run independently after the processed no-moisturizer dry-skin dataset has been generated.



## Output directories

The principal output folders are:

```text
outputs/
|-- Xy/
|-- training_results/
|-- roc_plots/
|-- cnn_ablation/
|-- pca/
|-- gradcam/
|-- perturbation_peak_trough/
|-- CNN_hydration_score/
`-- PC1_hydration_score/
```

The scripts create most output directories automatically.



## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/agrima-agarwal/ML_THz_SINATRA.git
cd ML_THz_SINATRA
pip install -r requirements.txt
```



## Dataset availability

Download the zip files from https://zenodo.org/records/17108141, extract the folders and paste them to the main repository



<<<<<<< HEAD
## Contact
=======
Run 'generate_Xy_{dataset}' (e.g. generate_Xy_skin-cancer) to read the data, perform signal pre-processing and generate the arrays X and y for ML model training or plots.  
Run 'Train models' to train the classical ML models and obtain their respective AUROC values.  
Run 'Train cnn models' to train 1D CNN and GradCAM.  
Run 'Make plots' to generate the plots for PCA and impulse function comparison.  

---

## Author
>>>>>>> 6962f2d1aa87e1eb4f45014ae99e20cee6f1be06

Agrima Agarwal  
University of Warwick  
agrima.agarwal@warwick.ac.uk



## Citation

If you use this code or dataset in your research, please cite:

A. Agarwal, et al. "Machine Learning Based In Vivo Classification of Skin Conditions with Terahertz Time-Domain Spectroscopy" 2026.




# ML-THz-SINATRA

Machine Learning–based analysis of Terahertz (THz) spectroscopy data for skin disease classification using the SINATRA dataset.
This project explores the use of **THz time-domain spectroscopy (THz-TDS)** for differentiating between **healthy, dry, and cancerous skin tissues** using machine learning models.
This repository contains all preprocessing, training and evaluation, and plotting scripts used for the experiments presented in our paper.


---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/agrima-agarwal/ML_THz_SINATRA.git
cd ML_THz_SINATRA
pip install -r requirements.txt
```

---

## Dataset availability

Download the zip files from https://zenodo.org/records/17108141, extract the folders and paste them to the main repository

---

## Usage

Run 'generate_Xy_{dataset}' (e.g. generate_Xy_skin-cancer) to read the data, perform signal pre-processing and generate the arrays X and y for ML model training or plots.  
Run 'Train models' to train the 10 defined ML models and obtain their respective AUROC values. Run 'Make plots' to generate the plots for PCA and impulse function comparisons as shown in the article.  

---

## Author

Agrima Agarwal  
University of Warwick  
agrima.agarwal@warwick.ac.uk  

---

## Citation

If you use this code or dataset in your research, please cite:

A. Agarwal, et al. "Terahertz Time-Domain Spectroscopy for Dermatologic Classification Using Machine Learning," 2025.





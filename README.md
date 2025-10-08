# ML-THz-SINATRA

Machine Learning–based analysis of Terahertz (THz) spectroscopy data for skin disease classification using the SINATRA dataset.
This project explores the use of **THz time-domain spectroscopy (THz-TDS)** for differentiating between **healthy, dry, and cancerous skin tissues** using machine learning models.
This repository contains all preprocessing, training, and evaluation scripts used for the experiments presented in our paper.

---


## Repository Structure

ML-THz-SINATRA/  
├── data/ # Processed datasets (ZIPs excluded)  
├── notebooks/ # Jupyter notebooks for analysis  
├── src/ # Core Python source code  
├── models/ # Trained model checkpoints  
├── results/ # Figures, metrics, and plots  
├── requirements.txt # List of dependencies  
└── README.md # You are here  



Start by downloading the data from : https://zenodo.org/records/17108141 \\
Unzip and copy the folders to the same directory Run 'generate_Xy_{dataset}' (e.g. generate_Xy_skin-cancer) to read the data, perform signal pre-processing and generate the arrays X and y for ML model training or plots.//
Run 'Train models' to train the 10 defined ML models and obtain their respective AUROC values. Run 'Make plots' to generate the plots for PCA and impulse function comparisons as shown in the article.


---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/agrima-agarwal/ML_THz_SINATRA.git
cd ML_THz_SINATRA
pip install -r requirements.txt

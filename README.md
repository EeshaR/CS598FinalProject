# CS598FinalProject

Project overview
This repo reproduces the temporal‑shift pipeline from Ji et al. (CHIL 2023) on the public Pima Indians Diabetes dataset. I create synthetic year labels (2016 and 2017), train logistic‑regression and decision‑tree models, measure performance drift, run a permutation test, discover subpopulations with high loss differences, and run an ablation on logistic‑regression regularization.

Files and what they do

data/diabetes.csv
the raw Pima dataset (≈100 KB). If the file is missing preprocessing.py will auto‑download it from the Plotly GitHub link.

src/preprocessing.py
loads diabetes.csv, drops rows with missing values, adds the synthetic year column, saves the cleaned file as data/diabetes_preprocessed.csv.

src/modeling.py
trains a logistic‑regression model and a decision‑tree model on the 2016 cohort, saves both fitted models to outputs/models/.

src/evaluation.py
loads the 2016‑trained models, trains fresh 2017 models, evaluates AUC, log‑loss, ΔAUC, and a 500‑draw permutation p‑value on the 2017 test set, writes results to outputs/results/metrics.csv.

src/subpopulation.py
computes per‑sample loss differences between the 2016‑trained LR and the 2017‑trained LR, labels samples as shifted or stable, fits a shallow decision tree to explain which feature combinations predict drift, writes the tree rules to outputs/results/subpop_rules.txt.

src/ablation.py
sweeps logistic‑regression C values (0.01 0.1 1 10 100), logs the resulting AUC on the 2017 cohort, saves outputs/results/ablation_C_auc.csv.

.ipynb notebook 
a single notebook that runs the entire workflow (data prep, modeling, evaluation, subpopulation, ablation) in one place. I made this for the sake of convience for reproducing output without downloading the entire repo 

requirements.txt
python dependencies: pandas 1.5.2, numpy 1.23.5, scikit‑learn 1.2.2, matplotlib 3.7.1, joblib 1.2.0.

How to run
clone the repo
git clone https://github.com/yourusername/temporal-shift-repro.git
pip install -r requirements.txt

run the pipeline step by step
python src/preprocessing.py
python src/modeling.py
python src/evaluation.py
python src/subpopulation.py
python src/ablation.py

You can also directly download the .ipynb notebook and run it from an editor or Google Collab to see the output results. 

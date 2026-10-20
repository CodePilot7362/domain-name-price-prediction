# Domain Name Value Classification

University coursework for **CN7050 – Intelligent Systems**. A Python machine-learning pipeline that classifies domain names into five value categories using character-level text features and handcrafted domain attributes.

## Methods

- Character-level TF-IDF (3–5-grams)
- Handcrafted features: length, digits, hyphens, dots and top-level domain
- Models: Logistic Regression, Linear SVM and Random Forest
- Stratified 80/20 train/test split
- Logistic Regression hyperparameter tuning with 5-fold GridSearchCV
- Evaluation: accuracy, macro precision, macro recall, classification reports and confusion matrices

## Run locally

Requires Python 3.9+.

```bash
pip install -r requirements.txt
python train_models.py
```

Keep `coursework_data.csv` in the same folder as `train_models.py`. The script displays confusion matrix plots and prints model evaluation metrics. Execution may take some time due to model training and cross-validation.

## Files

- `train_models.py` — original coursework code with the local CSV file path changed to a portable path
- `coursework_data.csv` — coursework dataset
- `requirements.txt` — Python dependencies

## Academic context

This is an academic group coursework project. The source code and experimental approach are retained for portfolio purposes; only the dataset path was made portable. Model results are produced when the script is run.

# Execution Instructions

## Required Packages:

- `numpy`
- `pandas`
- `tqdm`
- `scikit-learn`

## Before Running the Script:

1. Copy the datasets `tracks.csv` and `features.csv` into the directory `data/fma_metadata/`.

2. If you want to run the hyperparameter-tuning, set training to True in the main.py file. Otherwise, the best
   parameters found during tuning are loaded from `best_parameters.txt` and the evaluation is run with these parameters.

## Execution:

1. Run `main.py` to start the program.
#!/usr/bin/env python
# coding: utf-8
"""
SGIMC — real-application cross-validation script (HPC / SLURM).

Outer CV : 5 trials × 10-fold pairwise CV  (R's doCrossValidationByPairwise)
Inner CV : 10-fold pairwise CV on outer training matrix, evaluating 3 folds
           (matches outer density: ~90% of 1's in inner training, same as
            ~90% of 1's in outer training)
Metrics  : AUPR, AUC, F1, Accuracy, Recall, Specificity, Precision
Parallel : SLURM array tasks over (trial, fold, hyperparam) combos
"""

import os
import sys
import gzip
import pickle
import warnings
import logging

import numpy as np

from tqdm import TqdmSynchronisationWarning

warnings.simplefilter("ignore", TqdmSynchronisationWarning)

# multi-threaded: OMP/MKL thread count controlled by SLURM --cpus-per-task

# ====== user paths ======
PATH_ROOT = "/work/sfan/projects/BiSSGL"
DRIMC_PATH = os.path.join(PATH_ROOT, "scripts/methods/DRIMC")

# --- dataset selection (override with env var DATASET) ---
DATASET = os.environ.get("DATASET", "cdataset")
PATH_DATA = os.path.join(PATH_ROOT, "datasets/realAnalysis", DATASET)
PATH_OUTPUT = os.path.join(
    PATH_ROOT, "outputs/results/realAnalysis", DATASET, "fold_cv/origin"
)
PATH_ARCHIVE = os.path.join(PATH_OUTPUT, "archived")

for p in (PATH_OUTPUT, PATH_ARCHIVE):
    os.makedirs(p, exist_ok=True)

filenames = {"output": "results_sgimc"}

# ====== Python imports ======
sys.path.append(PATH_ROOT)

from sgimc.utils import mc_split, get_submatrix, load, save, sparsify_with_mask
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import ParameterGrid, ShuffleSplit
from scipy.sparse import coo_matrix

from sgimc import SparseGroupIMCClassifier

# ====== R setup via rpy2 (for CV splitting only) ======
import rpy2.robjects as robjects

r = robjects.r

r(f'setwd("{DRIMC_PATH}")')

# attach R packages — library() (not require) so sourced helpers see unqualified functions
for pkg in ["matrixcalc", "data.table", "Rcpp", "ROCR", "Bolstad2", "MESS"]:
    r(f"library({pkg})")

# source only the CV helper
r('source("doCrossValidationByPairwise.R")')
doCrossValidationByPairwise_R = r["doCrossValidationByPairwise"]


# ====== scoring helper (matches simulation scripts) ======
def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten())))
    )
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[
        np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)
    ]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(
        sorted(np.column_stack((recall_list, precision_list)).tolist())
    ).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]


# ====== helper: numpy matrix → R matrix ======
def to_r_matrix(arr):
    return r.matrix(
        robjects.FloatVector(arr.flatten()),
        byrow=True,
        nrow=arr.shape[0],
        ncol=arr.shape[1],
    )


# ====== helper: extract fold data from R savedFolds ======
def extract_fold(savedFolds, trial, fold):
    """
    Extract training matrix, test positions, and test labels from R CV output.

    Returns
    -------
    Y_train  : ndarray (n, m)  dense 0/1 training matrix (test entries zeroed)
    test_row : ndarray (T,)    0-based row indices of test entries
    test_col : ndarray (T,)    0-based col indices of test entries
    test_label : ndarray (T,)  0/1 test labels
    known_drug_idx : ndarray   1-based indices of drugs appearing in training
    known_target_idx : ndarray 1-based indices of targets appearing in training
    """
    fold_data = savedFolds.rx2(trial + 1).rx2(fold + 1)  # R is 1-indexed
    Y_train = np.array(fold_data.rx2(7))
    test_label = np.array(fold_data.rx2(1)).flatten()
    test_row = np.array(fold_data.rx2(3)).flatten().astype(int) - 1  # → 0-based
    test_col = np.array(fold_data.rx2(4)).flatten().astype(int) - 1
    known_drug_idx = np.array(fold_data.rx2(5)).flatten().astype(int)
    known_target_idx = np.array(fold_data.rx2(6)).flatten().astype(int)
    return Y_train, test_row, test_col, test_label, known_drug_idx, known_target_idx


# ====== load dataset ======
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
logging.info(
    "Threads: SLURM_CPUS_PER_TASK=%s, n_threads=%d",
    os.environ.get("SLURM_CPUS_PER_TASK"),
    n_threads,
)
logging.info("Loading dataset: %s", DATASET)

filename_input = os.path.join(PATH_DATA, "cv_data", "staged_dataset.gz")
U, V, Y = load(filename_input)

# U, V naming convention for BiSSGL (side-feature matrices)
U = np.asarray(U.toarray() if hasattr(U, "toarray") else U, dtype=float)
V = np.asarray(V.toarray() if hasattr(V, "toarray") else V, dtype=float)

logging.info(
    "Loaded U: %s,  V: %s",
    U.shape,
    V.shape,
)

# adjacency matrix for R's CV function (dense 0/1) — handle both sparse/dense inputs
Y_dense = Y.toarray() if hasattr(Y, "toarray") else np.asarray(Y)
Y_adj = (Y_dense > 0).astype(float)

# ====== create outer CV folds via R ======
kfold = 10
numSplit = 5
seeds_list = [7771, 8367, 22, 1812, 4659]
seeds_R = robjects.IntVector(seeds_list)

logging.info(
    "Creating %d trials × %d-fold pairwise CV splits via R ...", numSplit, kfold
)
savedFolds = doCrossValidationByPairwise_R(
    to_r_matrix(Y_adj), kfold=kfold, numSplit=numSplit, seeds=seeds_R
)

# ====== parameter grid ======
grid_model = ParameterGrid(
    {
        "C_lasso": [1.0, 1e-1],
        "C_group": [1.0, 1e-2, 1e-4],
        "C_ridge": [1.0],
        "rank": [50],
    }
)

# inner CV settings
N_INNER_KFOLD = 10  # inner CV folds — 10-fold matches outer density (~90% 1's in train)
N_INNER_EVAL = 3  # only evaluate first 3 of N_INNER_KFOLD folds to save time

# ====== flatten combinations into a list ======
combos = []
for trial in range(numSplit):
    for fold in range(kfold):
        for i_m, par_mdl in enumerate(grid_model):
            combos.append((trial, fold, i_m, par_mdl))
n_combos = len(combos)

# ====== SLURM array task id handling ======
task_id_env = os.environ.get("SLURM_ARRAY_TASK_ID") or os.environ.get("TASK_ID")
if task_id_env is None:
    if len(sys.argv) > 1:
        task_id_env = sys.argv[1]
    else:
        task_id_env = "0"

try:
    TASK_ID = int(task_id_env)
except ValueError:
    TASK_ID = 0

TASK_CHUNK_SIZE = int(os.environ.get("TASK_CHUNK_SIZE", "1"))
BASE_SEED = int(os.environ.get("BASE_SEED", str(0x0BADCAFE)))

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [task {TASK_ID}] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info(
    "Starting SLURM task %d (chunk size %d). Total combos: %d",
    TASK_ID,
    TASK_CHUNK_SIZE,
    n_combos,
)

start_idx = TASK_ID * TASK_CHUNK_SIZE
end_idx = min(start_idx + TASK_CHUNK_SIZE, n_combos)
idxs_to_run = range(start_idx, end_idx)

if start_idx >= n_combos:
    logging.info("No combos assigned to this task. Exiting.")
    sys.exit(0)

task_results = []

# cache inner CV folds per outer (trial, fold) — shared across HP combos
_inner_folds_cache = {}

# ====== run assigned combos ======
for combo_idx in idxs_to_run:
    trial, fold, i_m, par_mdl = combos[combo_idx]
    logging.info(
        "Running combo_idx=%d (trial=%d, fold=%d, rank=%d, "
        "C_lasso=%s, C_group=%s, C_ridge=%s)",
        combo_idx,
        trial,
        fold,
        par_mdl["rank"],
        par_mdl["C_lasso"],
        par_mdl["C_group"],
        par_mdl["C_ridge"],
    )

    # ------------------------------------------------------------------
    # Split RNG: depends only on (trial, fold) so that the same balanced
    # neg sample / inner splits are used across all HP combos.
    # ------------------------------------------------------------------
    model_seed = BASE_SEED + combo_idx

    try:
        # ====== extract fold data ======
        Y_train_dense, test_row, test_col, test_label, _, _ = extract_fold(
            savedFolds, trial, fold
        )

        # extract model params
        C_lasso = par_mdl["C_lasso"]
        C_group = par_mdl["C_group"]
        C_ridge = par_mdl["C_ridge"]
        rank = par_mdl["rank"]

        # ====== fit on full training → test scores ======
        # SGIMC uses {-1, +1} encoding: 1→+1, 0→-1
        Y_train_signed = Y_train_dense.copy()
        Y_train_signed[Y_train_signed == 0] = -1
        Y_train_sp = coo_matrix(Y_train_signed)

        model = SparseGroupIMCClassifier(
            rank,
            n_threads=n_threads,
            random_state=model_seed,
            C_lasso=C_lasso,
            C_group=C_group,
            C_ridge=C_ridge,
        )
        model.fit(U, V, Y_train_sp)

        prob_full = model.predict_proba(U, V)
        if hasattr(prob_full, "toarray"):
            prob_full = prob_full.toarray()
        prob_test = prob_full[test_row, test_col]
        scores_test = get_metrics(
            np.asarray(test_label, dtype=float),
            np.asarray(prob_test, dtype=float),
        )
        d1_test = int(sum(abs(model.coef_W_).max(axis=1) > 0))
        d2_test = int(sum(abs(model.coef_H_).max(axis=1) > 0))
        # selected features: {index: L2 row norm} for non-zero rows of W and H
        _sel_W = np.where(abs(model.coef_W_).max(axis=1) > 0)[0]
        _sel_H = np.where(abs(model.coef_H_).max(axis=1) > 0)[0]
        sel_W = {int(i): float(np.linalg.norm(model.coef_W_[i])) for i in _sel_W}
        sel_H = {int(i): float(np.linalg.norm(model.coef_H_[i])) for i in _sel_H}

        # ====== inner CV — pairwise CV on outer training matrix ======
        # Cache inner folds per (trial, fold): same split for all HP combos
        inner_cache_key = (trial, fold)
        if inner_cache_key not in _inner_folds_cache:
            inner_seed = (BASE_SEED + trial * 100 + fold + 31415) % (2**31 - 1)
            logging.info(
                "Creating inner %d-fold CV (evaluating %d) for trial=%d, fold=%d (seed=%d)",
                N_INNER_KFOLD,
                N_INNER_EVAL,
                trial,
                fold,
                inner_seed,
            )
            inner_savedFolds = doCrossValidationByPairwise_R(
                to_r_matrix(Y_train_dense),
                kfold=N_INNER_KFOLD,
                numSplit=1,
                seeds=robjects.IntVector([inner_seed]),
            )
            _inner_folds_cache[inner_cache_key] = inner_savedFolds
        inner_savedFolds = _inner_folds_cache[inner_cache_key]

        for cv in range(N_INNER_EVAL):
            # extract inner fold: Y_inner has validation entries zeroed out
            Y_inner, val_row, val_col, val_label, _, _ = extract_fold(
                inner_savedFolds, 0, cv  # trial=0 since numSplit=1
            )

            # convert to SGIMC encoding
            Y_inner_signed = Y_inner.copy()
            Y_inner_signed[Y_inner_signed == 0] = -1
            Y_inner_sp = coo_matrix(Y_inner_signed)

            model_cv = SparseGroupIMCClassifier(
                rank,
                n_threads=n_threads,
                random_state=model_seed,
                C_lasso=C_lasso,
                C_group=C_group,
                C_ridge=C_ridge,
            )
            model_cv.fit(U, V, Y_inner_sp)

            prob_full_cv = model_cv.predict_proba(U, V)
            if hasattr(prob_full_cv, "toarray"):
                prob_full_cv = prob_full_cv.toarray()
            prob_valid = prob_full_cv[val_row, val_col]
            scores_valid = get_metrics(
                np.asarray(val_label, dtype=float),
                np.asarray(prob_valid, dtype=float),
            )
            d1_valid = int(sum(abs(model_cv.coef_W_).max(axis=1) > 0))
            d2_valid = int(sum(abs(model_cv.coef_H_).max(axis=1) > 0))

            task_results.append(
                {
                    # --- fold identity ---
                    "trial": trial,
                    "fold": fold,
                    # --- hyperparameters ---
                    "C_lasso": C_lasso,
                    "C_group": C_group,
                    "C_ridge": C_ridge,
                    "rank": rank,
                    # --- inner CV fold index ---
                    "cv": int(cv),
                    # --- validation scores (for HP selection) ---
                    "val_score": scores_valid,
                    "val_d1": d1_valid,
                    "val_d2": d2_valid,
                    # --- test scores (reported after HP selection) ---
                    "test_score": scores_test,
                    "test_d1": d1_test,
                    "test_d2": d2_test,
                    # --- selected features: {idx: L2_norm} (test model) ---
                    "test_sel_W": sel_W,
                    "test_sel_H": sel_H,
                }
            )

    except Exception as e:
        logging.exception("Error at combo_idx=%d: %s", combo_idx, str(e))

# ====== save task results ======
outfile = os.path.join(PATH_OUTPUT, f"{filenames['output']}_task{TASK_ID}.gz")
logging.info("Saving %d result rows to %s", len(task_results), outfile)
with gzip.open(outfile, "wb+", 4) as fout:
    pickle.dump(task_results, fout)

logging.info("Task %d finished.", TASK_ID)

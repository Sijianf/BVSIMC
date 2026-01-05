#!/usr/bin/env python
# coding: utf-8

# # Run BiSSGL

# In[1]:


import os
import time
import gzip
import pickle
import warnings

# import matplotlib.pyplot as plt

import numpy as np

from tqdm import TqdmSynchronisationWarning

warnings.simplefilter("ignore", TqdmSynchronisationWarning)


# In[2]:


PATH_ROOT = "/Users/sijianfan/Documents/projects/BiSSGL"
PATH_DATA = os.path.join(PATH_ROOT, "datasets/realAnalysis/tuberculosis/cv_data")

PATH_OUTPUT = os.path.join(PATH_ROOT, "outputs/realAnalysis/tuberculosis/")
if not os.path.isdir(PATH_OUTPUT):
    os.makedirs(PATH_OUTPUT)

PATH_ARCHIVE = os.path.join(PATH_OUTPUT, "archived")
if not os.path.isdir(PATH_ARCHIVE):
    os.makedirs(PATH_ARCHIVE)


# In[3]:


filename_staged = os.path.join(PATH_DATA, "staged_dataset.gz")

filenames = {"input": "staged_dataset.gz", "output": "results_bissgl.gz"}


# In[4]:

filename_input = os.path.join(PATH_DATA, filenames["input"])

filename_output = os.path.join(PATH_OUTPUT, filenames["output"])

if os.path.exists(filename_output):
    mdttm = time.strftime("%Y%m%d_%H%M%S")
    os.rename(
        filename_output,
        os.path.join(PATH_ARCHIVE, "%s%s" % (mdttm, filenames["output"])),
    )


# In[5]:

from sgimc.utils import mc_split


# In[6]:

from sgimc.utils import get_submatrix


# In[7]:

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sgimc.utils import sparsify_with_mask


def mc_get_scores(R_true, R_prob):
    R_pred = np.where(R_prob.data > 0.5, 1, -1)

    # compute the confusion matrix for ±1 labels (`-1` is negative)
    ii, jj = ((R_pred + 1) // 2).astype(int), ((R_true.data + 1) // 2).astype(int)
    cnfsn = confusion_matrix(y_true=jj, y_pred=ii)

    return {
        "tn": cnfsn[0, 0],
        "fn": cnfsn[1, 0],
        "fp": cnfsn[0, 1],
        "tp": cnfsn[1, 1],
        "auc": roc_auc_score(R_true.data, R_prob.data),
    }


# In[8]:


random_state = np.random.RandomState(0x0BADCAFE)


# In[72]:


from sgimc.utils import load, save

U, V, Y = load(filename_input)
Y[Y == -1] = 0  # transfer Y from {-1, 1} to {0, 1}.


# In[10]:

dvlp_size, test_size = 0.9, 0.1

ind_dvlp, ind_test = next(
    mc_split(
        Y,
        n_splits=1,
        random_state=random_state,
        train_size=dvlp_size,
        test_size=test_size,
    )
)

Y_test = get_submatrix(Y, ind_test)


# In[11]:


from sklearn.model_selection import ParameterGrid

grid_dataset = ParameterGrid(
    {
        "train_size": np.arange(0.01, 0.1, 0.02),
        "n_splits": [3],
    }
)

grid_model = ParameterGrid(
    {
        "tilde_lambda0": [1, 3, 5],
        "lambda0": [1, 3, 5],
        "xi": [1, 2, 3, 5, 10],
        "eta": [1e-6, 1e-8],
        "K": [3, 6, 13],
    }
)


# In[12]:


import sys

sys.path.append(PATH_ROOT)

from scripts.methods.BiSSGL.BiSSGLc import BiSSGL


from tqdm import tqdm
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import train_test_split
from scipy.special import expit

import itertools
import multiprocessing as mp
from joblib import Parallel, delayed


# --- tune these ---
N_JOBS = min(max(1, os.cpu_count() - 1), 16)  # adjust as you like
RANDOM_STATE = random_state  # reuse your existing random_state
# ---------------------

# try to use forkserver on Linux to reduce forking issues (safe with loky)
try:
    mp.set_start_method("forkserver")
except RuntimeError:
    # already set or not supported; ok to continue
    pass

# build task list of (par_dtst, par_mdl)
tasks = [(par_dtst, par_mdl) for par_dtst in grid_dataset for par_mdl in grid_model]


def worker(par_dtst, par_mdl):
    """Run the original logic for one (dataset-config, model-config) pair.
    Returns a list of result dicts (one per CV fold)."""
    local_results = []

    # prepare train indices: same logic as original (take prefix by train_size)
    ind_train_all, _ = train_test_split(
        ind_dvlp,
        shuffle=False,
        random_state=RANDOM_STATE,
        test_size=(1 - (par_dtst["train_size"] / dvlp_size)),
    )

    # full-train on development set (as in your original code)
    Y_train_full = get_submatrix(Y, ind_train_all)

    xi = par_mdl["xi"]
    eta = par_mdl["eta"]
    tilde_lambda0 = par_mdl["tilde_lambda0"]
    lambda0 = par_mdl["lambda0"]
    K = par_mdl["K"]

    # model on whole development dataset
    model = BiSSGL(
        Y=Y_train_full.toarray(),
        U=U,
        V=V,
        xi=xi,
        eta=eta,
        tilde_lambda0=tilde_lambda0,
        tilde_lambda1=1,
        tilde_alpha=1 / K,
        tilde_beta=1,
        lambda0=lambda0,
        lambda1=1,
        alpha=1 / K,
        beta=1,
        K=K,
        max_iter=1000,
        tol=1e-8,
    )
    est_mu, est_A, est_B, logLik = model.optimization()

    # test metrics (single, outside CV loop)
    prob_full = expit(U @ est_A @ est_B.T @ V.T)
    prob_test = get_submatrix(prob_full, ind_test)
    scores_test = mc_get_scores(Y_test, prob_test)
    d1_test = int((abs(est_A).max(axis=1) > 0).sum())
    d2_test = int((abs(est_B).max(axis=1) > 0).sum())

    # run k-fold CV on ind_train_all
    splt = KFold(par_dtst["n_splits"], shuffle=True, random_state=RANDOM_STATE)
    for cv, (ind_train_idx, ind_valid_idx) in enumerate(splt.split(ind_train_all)):
        ind_train = ind_train_all[ind_train_idx]
        ind_valid = ind_train_all[ind_valid_idx]

        Y_train = get_submatrix(Y, ind_train)
        Y_valid = get_submatrix(Y, ind_valid)

        model_cv = BiSSGL(
            Y=Y_train.toarray(),
            U=U,
            V=V,
            xi=xi,
            eta=eta,
            tilde_lambda0=tilde_lambda0,
            tilde_lambda1=1,
            tilde_alpha=1 / K,
            tilde_beta=1,
            lambda0=lambda0,
            lambda1=1,
            alpha=1 / K,
            beta=1,
            K=K,
            max_iter=1000,
            tol=1e-8,
        )
        est_mu_cv, est_A_cv, est_B_cv, logLik_cv = model_cv.optimization()

        prob_full_cv = expit(U @ est_A_cv @ est_B_cv.T @ V.T)
        prob_valid = get_submatrix(prob_full_cv, ind_valid)
        scores_valid = mc_get_scores(Y_valid, prob_valid)
        d1_valid = int((abs(est_A_cv).max(axis=1) > 0).sum())
        d2_valid = int((abs(est_B_cv).max(axis=1) > 0).sum())

        local_results.append(
            {
                "train_size": par_dtst["train_size"],
                "xi": par_mdl["xi"],
                "eta": par_mdl["eta"],
                "tilde_lambda0": par_mdl["tilde_lambda0"],
                "lambda0": par_mdl["lambda0"],
                "K": par_mdl["K"],
                "cv": cv,
                "val_score": scores_valid["auc"],
                "val_d1": d1_valid,
                "val_d2": d2_valid,
                "test_score": scores_test["auc"],
                "test_d1": d1_test,
                "test_d2": d2_test,
            }
        )

    return local_results


# run in parallel; chunk size reduces pickling overhead
n_tasks = len(tasks)
chunksize = max(1, n_tasks // (N_JOBS * 4))

parallel = Parallel(n_jobs=N_JOBS, backend="loky", prefer="processes", verbose=10)
# produce list of lists, then flatten
all_results_nested = parallel(
    delayed(worker)(par_dtst, par_mdl) for (par_dtst, par_mdl) in tasks
)
results = [r for sub in all_results_nested for r in sub]

# Save results
with gzip.open(filename_output, "wb+", 4) as fout:
    pickle.dump(results, fout)

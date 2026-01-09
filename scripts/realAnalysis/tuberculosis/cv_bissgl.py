#!/usr/bin/env python
# coding: utf-8

# # Run BiSSGL

# In[1]:

import os
import time
import gzip
import pickle
import warnings

import numpy as np

from tqdm import TqdmSynchronisationWarning

warnings.simplefilter("ignore", TqdmSynchronisationWarning)


# In[2]:

PATH_ROOT = "/work/sfan/projects/BiSSGL"
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
        "tilde_lambda0": [1, 3, 5, 10, 50],
        "lambda0": [1, 3, 5, 10, 50],
        "xi": [1, 2, 4, 6, 8, 10],
        "eta": [1e-6, 1e-8, 1e-10],
        "K": [3, 6, 13],
    }
)


# In[12]:


import sys

sys.path.append(PATH_ROOT)

from scripts.methods.BiSSGL.BiSSGLc import BiSSGL


from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from scipy.special import expit

results = []
for par_dtst in tqdm(grid_dataset):
    # prepare the train dataset: take the specified share from the beginnig of the index array
    ind_train_all, _ = train_test_split(
        ind_dvlp,
        shuffle=False,
        random_state=random_state,
        test_size=(1 - (par_dtst["train_size"] / dvlp_size)),
    )

    # Run the experiment: the model
    for par_mdl in grid_model:  # tqdm.tqdm(, desc="cv %02d" % (cv,))
        Y_train = get_submatrix(Y, ind_train_all)

        # set up the model
        xi, eta, tilde_lambda0, lambda0, K = (
            par_mdl["xi"],
            par_mdl["eta"],
            par_mdl["tilde_lambda0"],
            par_mdl["lambda0"],
            par_mdl["K"],
        )

        model = BiSSGL(
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

        # fit on the whole development dataset
        est_mu, est_A, est_B, logLik = model.optimization()

        # get the score
        prob_full = expit(U @ est_A @ est_B.T @ V.T)
        prob_test = get_submatrix(prob_full, ind_test)
        scores_test = mc_get_scores(Y_test, prob_test)
        d1_test = sum(abs(est_A).max(axis=1) > 0)
        d2_test = sum(abs(est_B).max(axis=1) > 0)

        # run the k-fold CV
        # splt = ShuffleSplit(**par_dtst, random_state=random_state)
        splt = KFold(par_dtst["n_splits"], shuffle=True, random_state=random_state)
        for cv, (ind_train, ind_valid) in enumerate(splt.split(ind_train_all)):

            # prepare the train and test indices
            ind_train, ind_valid = ind_train_all[ind_train], ind_train_all[ind_valid]
            Y_train = get_submatrix(Y, ind_train)
            Y_valid = get_submatrix(Y, ind_valid)

            # fit the model
            model = BiSSGL(
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

            # fit on the cv dataset
            est_mu, est_A, est_B, logLik = model.optimization()

            # compute the class probabilities
            prob_full = expit(U @ est_A @ est_B.T @ V.T)
            prob_valid = get_submatrix(prob_full, ind_valid)
            scores_valid = mc_get_scores(Y_valid, prob_valid)
            d1_valid = sum(abs(est_A).max(axis=1) > 0)
            d2_valid = sum(abs(est_B).max(axis=1) > 0)

            # record the results
            results.append(
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
        # end for
    # end for
# end for

# Save the results in a pickle

with gzip.open(filename_output, "wb+", 4) as fout:
    pickle.dump(results, fout)

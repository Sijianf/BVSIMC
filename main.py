# This is a Python file for binary inductive matrix completion using spike-and-slab group lasso prior.
# Author: Sijian Fan
# Date: June 2025

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_data(
    num_users, num_items, num_features, sparsity=0.8, random_state=42
):
    np.random.seed(random_state)
    user_features = np.random.randn(num_users, num_features)
    item_features = np.random.randn(num_items, num_features)

    true_weights = np.random.randn(num_features, num_features)
    interaction_matrix = user_features @ true_weights @ item_features.T
    interaction_matrix = (interaction_matrix > 0).astype(int)  # Binary interactions

    mask = np.random.rand(num_users, num_items) > sparsity
    interaction_matrix *= mask

    return interaction_matrix, user_features, item_features


def train_test_split_interactions(interaction_matrix, test_size=0.2, random_state=42):
    user_item_pairs = np.array(interaction_matrix.nonzero()).T
    train_pairs, test_pairs = train_test_split(
        user_item_pairs, test_size=test_size, random_state=random_state
    )

    train_matrix = sp.coo_matrix(
        (np.ones(len(train_pairs)), (train_pairs[:, 0], train_pairs[:, 1])),
        shape=interaction_matrix.shape,
    ).toarray()
    test_matrix = sp.coo_matrix(
        (np.ones(len(test_pairs)), (test_pairs[:, 0], test_pairs[:, 1])),
        shape=interaction_matrix.shape,
    ).toarray()

    return train_matrix, test_matrix


def fit_logistic_regression(train_matrix, user_features, item_features):
    num_users, num_items = train_matrix.shape
    X = []
    y = []

    for i in range(num_users):
        for j in range(num_items):
            if train_matrix[i, j] != 0:
                feature_vector = np.concatenate([user_features[i], item_features[j]])
                X.append(feature_vector)
                y.append(train_matrix[i, j])

    X = np.array(X)
    y = np.array(y)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model


def evaluate_model(model, test_matrix, user_features, item_features):
    num_users, num_items = test_matrix.shape
    X_test = []
    y_test = []

    for i in range(num_users):
        for j in range(num_items):
            if test_matrix[i, j] != 0:
                feature_vector = np.concatenate([user_features[i], item_features[j]])
                X_test.append(feature_vector)
                y_test.append(test_matrix[i, j])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    y_scores = model.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, y_scores)
    auc_pr = average_precision_score(y_test, y_scores)

    return auc_roc, auc_pr


def main():
    num_users = 100
    num_items = 100
    num_features = 10

    interaction_matrix, user_features, item_features = generate_synthetic_data(
        num_users, num_items, num_features
    )
    train_matrix, test_matrix = train_test_split_interactions(interaction_matrix)

    model = fit_logistic_regression(train_matrix, user_features, item_features)

    auc_roc, auc_pr = evaluate_model(model, test_matrix, user_features, item_features)

    logger.info(f"AUC-ROC: {auc_roc:.4f}")
    logger.info(f"AUC-PR: {auc_pr:.4f}")


if __name__ == "__main__":
    main()

# This is a Python file for binary inductive matrix completion using spike-and-slab group lasso prior.
# Author: Sijian Fan
# Date: November 03 2025

import numpy as np
from scipy import linalg
from scipy.special import gamma, expit


def group_lasso_density(vec, lambda_):
    size = len(vec)
    norm = linalg.norm(vec)
    density = (
        2 ** (-size)
        * np.pi ** (-(size - 1) / 2)
        / gamma((size + 1) / 2)
        * lambda_**size
        * np.exp(-norm * lambda_)
    )
    return density


def p_star(vec, theta, lambda0, lambda1):
    spike = (1 - theta) * group_lasso_density(vec, lambda0)
    slab = theta * group_lasso_density(vec, lambda1)
    # p_star_value = slab / (spike + slab)
    if np.isnan(spike) or np.isnan(slab):
        p_star_value = np.nan
    elif spike + slab == 0:
        p_star_value = 0  # or whatever makes sense
    else:
        p_star_value = slab / (spike + slab)
    return p_star_value


def lambda_star(vec, theta, lambda0, lambda1):
    p_star_value = p_star(vec, theta, lambda0, lambda1)
    lambda_star_value = (1 - p_star_value) * lambda0 + p_star_value * lambda1
    return lambda_star_value


def update_momentum(x, x_lag, iter):
    momentum = x + (iter - 2) / (iter + 1) * (x - x_lag)
    return momentum


def h_function(vec, theta, lambda0, lambda1, eta):
    p_star_value = p_star(vec, theta, lambda0, lambda1)
    lambda_star_value = lambda_star(vec, theta, lambda0, lambda1)
    h_value = (lambda_star_value - lambda1) ** 2 + 2 / eta * np.log(p_star_value)
    return h_value


def update_delta(size, theta, lambda0, lambda1, eta):
    zero_vec = np.zeros(size)
    p_star_value = p_star(zero_vec, theta, lambda0, lambda1)
    h_value = h_function(zero_vec, theta, lambda0, lambda1, eta)
    if h_value > 0:
        delta = np.sqrt(2 * eta * np.log(1 / p_star_value)) + eta * lambda1
    else:
        delta = eta * lambda_star(zero_vec, theta, lambda0, lambda1)
    return delta


def update_theta(mat, alpha, beta):
    count = 0
    nrow = mat.shape[0]
    for i in range(nrow):
        if linalg.norm(mat[i, :], 0) != 0:
            count += 1
    theta = (alpha + count) / (alpha + beta + nrow)
    return theta


def get_W(Y, mu, U, V, A, B, xi):
    W = (xi * Y + 1 - Y) * expit(np.outer(mu, np.ones(Y.shape[1])) + U @ A @ B.T @ V.T)
    return W


def gradient(side, Y, mu, U, V, A, B, xi):
    W = get_W(Y, mu, U, V, A, B, xi)
    if side == "A":
        grad = U.T @ (W - xi * Y) @ V @ B
    elif side == "B":
        grad = V.T @ (W - xi * Y).T @ U @ A
    return grad


def SSGL(vec, theta, z, delta, eta, lambda0, lambda1):
    if linalg.norm(z) <= delta:
        return np.zeros(vec.shape)
    else:
        temp = 1 - eta * lambda_star(vec, theta, lambda0, lambda1) / linalg.norm(z)
        if temp > 0:
            return temp * z
        else:
            return np.zeros(vec.shape)


def log_likelihood(
    Y,
    mu,
    U,
    V,
    A,
    B,
    xi,
    theta,
    lambda0,
    lambda1,
    tilde_theta,
    tilde_lambda0,
    tilde_lambda1,
):
    M = np.outer(mu, np.ones(Y.shape[1])) + U @ A @ B.T @ V.T
    d1 = U.shape[1]
    d2 = V.shape[1]
    LambdaStarA = np.diag(
        [
            lambda_star(A[i, :], tilde_theta, tilde_lambda0, tilde_lambda1)
            for i in range(d1)
        ]
    )
    LambdaStarB = np.diag(
        [lambda_star(B[j, :], theta, lambda0, lambda1) for j in range(d2)]
    )
    penalty = np.sum(np.dot(LambdaStarA, A)) + np.sum(np.dot(LambdaStarB, B))

    loglik = np.sum(xi * Y * M - (xi * Y + 1 - Y) * np.log(1 + np.exp(M))) - penalty
    return loglik


def update_mu(Y, mu, U, V, A, B, xi):
    M = np.outer(mu, np.ones(Y.shape[1])) + U @ A @ B.T @ V.T
    P = expit(M)
    for i in range(len(mu)):
        mu[i] = (
            mu[i]
            + 4
            / (xi * Y[i, :] + 1 - Y[i, :]).sum()
            * (xi * Y[i, :] - (xi * Y[i, :] + 1 - Y[i, :]) * P[i, :]).sum()
        )
    return mu


def optimization(
    Y,
    U,
    V,
    xi,
    eta,
    tilde_lambda0,
    tilde_lambda1,
    tilde_alpha,
    tilde_beta,
    lambda0,
    lambda1,
    alpha,
    beta,
    K=None,
    mu=None,
    A=None,
    B=None,
    seed=None,
    max_iter=100,
    tol=1e-3,
):

    d1 = U.shape[1]
    d2 = V.shape[1]

    # Initialize K
    if K is None:
        if A is None and B is None:
            K = len(np.linalg.svd(Y).S)
        else:
            ncol_a = A.shape[1] if A is not None else 0
            ncol_b = B.shape[1] if B is not None else 0
        K = max(ncol_a, ncol_b)

    # Initialize A and B
    if A is None:
        if seed is not None:
            np.random.seed(seed)
            A = np.sqrt(1 / K) * np.random.normal(size=(d1, K))
        else:
            A = np.sqrt(1 / K) * np.random.normal(size=(d1, K))
    if B is None:
        if seed is not None:
            np.random.seed(seed)
            B = np.sqrt(1 / K) * np.random.normal(size=(d2, K))
        else:
            B = np.sqrt(1 / K) * np.random.normal(size=(d2, K))

    # Initialize mu
    if mu is None:
        mu = np.zeros(Y.shape[0])

    # Initialize theta
    tilde_theta = 0.5
    theta = 0.5

    # Initialize delta
    tilde_delta = update_delta(K, tilde_theta, tilde_lambda0, tilde_lambda1, eta)
    delta = update_delta(K, theta, lambda0, lambda1, eta)

    # Initialize momentem
    A_lag = A.copy()
    B_lag = B.copy()
    A_momentum = update_momentum(A, A_lag, 2)
    B_momentum = update_momentum(B, B_lag, 2)

    # Monitor log likelihood
    logLik = []

    # Main iteration
    for iter in range(2, max_iter):
        logLik.append(
            log_likelihood(
                Y,
                mu,
                U,
                V,
                A,
                B,
                xi,
                theta,
                lambda0,
                lambda1,
                tilde_theta,
                tilde_lambda0,
                tilde_lambda1,
            )
        )
        # update A
        A_momentum = update_momentum(A, A_lag, iter)
        grad_A = gradient("A", Y, mu, U, V, A_momentum, B_momentum, xi)
        # print(grad_A[0])
        tilde_Z = A_momentum - eta * grad_A
        A_lag = A.copy()
        for i in range(d1):
            A[i, :] = SSGL(
                A_momentum[i, :],
                tilde_theta,
                tilde_Z[i, :],
                tilde_delta,
                eta,
                tilde_lambda0,
                tilde_lambda1,
            )

        # update B
        B_momentum = update_momentum(B, B_lag, iter)
        grad_B = gradient("B", Y, mu, U, V, A_momentum, B_momentum, xi)
        # print(grad_B[0])
        Z = B_momentum - eta * grad_B
        B_lag = B.copy()
        for j in range(d2):
            B[j, :] = SSGL(
                B_momentum[j, :],
                theta,
                Z[j, :],
                delta,
                eta,
                lambda0,
                lambda1,
            )

        # Update theta and delta
        if iter % 10 == 0:
            tilde_theta = update_theta(A, tilde_alpha, tilde_beta)
            theta = update_theta(B, alpha, beta)
            tilde_delta = update_delta(
                K, tilde_theta, tilde_lambda0, tilde_lambda1, eta
            )
            delta = update_delta(K, theta, lambda0, lambda1, eta)

        # Update mu
        mu = update_mu(Y, mu, U, V, A_momentum, B_momentum, xi) * 0

        # check convergence
        norm_A = linalg.norm(A - A_lag) / (linalg.norm(A_lag) + 1e-8)
        norm_B = linalg.norm(B - B_lag) / (linalg.norm(B_lag) + 1e-8)
        if max(norm_A, norm_B) < tol:
            break

    print(f"Finished with iterations of {iter}")

    return mu, A, B, logLik

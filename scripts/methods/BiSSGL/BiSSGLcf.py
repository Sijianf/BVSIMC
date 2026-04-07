# BiSSGL class version
# Binary inductive matrix completion with spike-and-slab group lasso prior
# 2026.02: Converted from top-level functions into an encapsulated class
# 2026.04: Updated codes for speeding up
# Author: Sijian Fan <sijianstats@gmail.com>

import numpy as np
from scipy import linalg
from scipy.special import gamma, expit


class BiSSGL:
    def __init__(
        self,
        Y,
        U,
        V,
        xi=1,
        eta=0.0001,
        tilde_lambda0=10,
        tilde_lambda1=1,
        tilde_alpha=1,
        tilde_beta=1,
        lambda0=10,
        lambda1=1,
        alpha=1,
        beta=1,
        K=None,
        max_iter=1000,
        tol=1e-5,
        shrink=0.5,
    ):
        # data / shapes
        self.Y = Y
        self.U = U
        self.V = V
        self.I, self.J = Y.shape

        # hyperparameters
        self.xi = xi
        self.eta = eta
        self.tilde_lambda0 = tilde_lambda0
        self.tilde_lambda1 = tilde_lambda1
        self.tilde_alpha = tilde_alpha
        self.tilde_beta = tilde_beta
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.alpha = alpha
        self.beta = beta

        # algorithm settings
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.shrink = shrink

        # cached quantities used many times
        self.Y_xi = self.xi * self.Y
        self.Y_weight = self.xi * self.Y + 1 - self.Y

        # model params (to be set / initialized in optimization)
        self.mu = None
        self.A = None
        self.B = None

    # ----- helper functions -----
    def _compute_UA_VB(self, A, B):
        UA = self.U @ A
        VB = self.V @ B
        return UA, VB

    def _compute_M(self, mu, A, B):
        UA, VB = self._compute_UA_VB(A, B)
        M = mu[:, None] + UA @ VB.T
        return M

    def _compute_M_from_UA_VB(self, mu, UA, VB):
        return mu[:, None] + UA @ VB.T

    def group_lasso_density(self, vec, lambda_):
        size = len(vec)
        norm = linalg.norm(vec)
        log_density = (
            -size * np.log(2)
            - (size - 1) / 2 * np.log(np.pi)
            - np.log(gamma((size + 1) / 2))
            + size * np.log(lambda_)
            - lambda_ * norm
        )
        density = np.exp(log_density)
        return density

    def p_star(self, vec, theta, lambda0, lambda1):
        # faster version of the same spike/slab ratio idea
        theta = np.clip(theta, np.finfo(float).tiny, 1 - np.finfo(float).eps)
        size = len(vec)
        norm = linalg.norm(vec)

        log_spike = np.log(1 - theta) + size * np.log(lambda0) - lambda0 * norm
        log_slab = np.log(theta) + size * np.log(lambda1) - lambda1 * norm

        m = max(log_spike, log_slab)
        spike = np.exp(log_spike - m)
        slab = np.exp(log_slab - m)

        if np.isnan(spike) or np.isnan(slab):
            return 1.0
        if spike + slab == 0:
            return 1.0
        return slab / (spike + slab)

    def lambda_star(self, vec, theta, lambda0, lambda1):
        p = self.p_star(vec, theta, lambda0, lambda1)
        return (1 - p) * lambda0 + p * lambda1

    def update_momentum(self, x, x_lag, iter_):
        return x + (iter_ - 2) / (iter_ + 1) * (x - x_lag)

    def h_function(self, vec, theta, lambda0, lambda1, eta):
        p = self.p_star(vec, theta, lambda0, lambda1)
        ls = self.lambda_star(vec, theta, lambda0, lambda1)
        p = max(p, np.finfo(float).tiny)
        return (ls - lambda1) ** 2 + 2 / eta * np.log(p)

    def update_delta(self, size, theta, lambda0, lambda1, eta):
        zero_vec = np.zeros(size)
        p = self.p_star(zero_vec, theta, lambda0, lambda1)
        h = self.h_function(zero_vec, theta, lambda0, lambda1, eta)
        if h > 0:
            return np.sqrt(2 * eta * np.log(1 / p)) + eta * lambda1
        else:
            return eta * self.lambda_star(zero_vec, theta, lambda0, lambda1)

    def update_theta(self, mat, alpha, beta):
        # same idea, vectorized
        count = np.count_nonzero(np.any(mat != 0, axis=1))
        return (alpha + count) / (alpha + beta + mat.shape[0])

    def get_W(self, mu, A, B):
        M = self._compute_M(mu, A, B)
        return self.Y_weight * expit(M)

    def gradient(self, side, mu, A, B):
        UA, VB = self._compute_UA_VB(A, B)
        M = self._compute_M_from_UA_VB(mu, UA, VB)
        W_minus = self.Y_weight * expit(M) - self.Y_xi

        if side == "A":
            return self.U.T @ W_minus @ VB
        elif side == "B":
            return self.V.T @ W_minus.T @ UA
        else:
            raise ValueError("side must be 'A' or 'B'")

    def SSGL(self, vec, theta, z, delta, eta, lambda0, lambda1):
        if not np.all(np.isfinite(z)):
            return np.zeros(vec.shape)
        z_norm = linalg.norm(z)
        if z_norm <= delta:
            return np.zeros(vec.shape)
        temp = 1 - eta * self.lambda_star(vec, theta, lambda0, lambda1) / z_norm
        return temp * z

    def log_likelihood(
        self,
        mu,
        A,
        B,
        theta,
        lambda0,
        lambda1,
        tilde_theta,
        tilde_lambda0,
        tilde_lambda1,
    ):
        M = self._compute_M(mu, A, B)
        loglik = np.sum(self.Y_xi * M - self.Y_weight * np.logaddexp(0, M))
        return loglik

    def obj_function(
        self,
        mu,
        A,
        B,
        theta,
        lambda0,
        lambda1,
        tilde_theta,
        tilde_lambda0,
        tilde_lambda1,
    ):
        M = self._compute_M(mu, A, B)
        d1 = self.U.shape[1]
        d2 = self.V.shape[1]

        LambdaStarA = np.diag(
            [
                self.lambda_star(A[i, :], tilde_theta, tilde_lambda0, tilde_lambda1)
                for i in range(d1)
            ]
        )
        LambdaStarB = np.diag(
            [self.lambda_star(B[j, :], theta, lambda0, lambda1) for j in range(d2)]
        )

        # kept as in your original idea
        penalty = np.sum(np.dot(LambdaStarA, A)) + np.sum(np.dot(LambdaStarB, B))

        loglik = np.sum(self.Y_xi * M - self.Y_weight * np.logaddexp(0, M)) - penalty
        return loglik

    def update_mu(self, mu, A, B):
        # kept original update idea; only matrix construction is faster
        M = self._compute_M(mu, A, B)
        P = expit(M)

        for i in range(len(mu)):
            denom = self.Y_weight[i, :].sum()
            if denom != 0:
                mu[i] = (
                    mu[i]
                    + 4
                    / denom
                    * (self.Y_xi[i, :] - self.Y_weight[i, :] * P[i, :]).sum()
                )
        return mu

    # ----- main optimization runner -----
    def optimization(
        self, K=None, mu=None, A=None, B=None, seed=None, max_iter=None, tol=None
    ):
        if max_iter is None:
            max_iter = self.max_iter
        if tol is None:
            tol = self.tol
        if K is None:
            K = self.K

        d1 = self.U.shape[1]
        d2 = self.V.shape[1]

        # initialize K if needed
        if K is None:
            if A is None and B is None:
                sv = np.linalg.svd(self.Y, compute_uv=False)
                K = len(sv)
            else:
                ncol_a = A.shape[1] if A is not None else 0
                ncol_b = B.shape[1] if B is not None else 0
                K = max(ncol_a, ncol_b)

        # initialize A, B
        if A is None:
            if seed is not None:
                np.random.seed(seed)
            A = np.sqrt(1 / K) * np.random.normal(size=(d1, K))

        if B is None:
            if seed is not None:
                np.random.seed(seed)
            B = np.sqrt(1 / K) * np.random.normal(size=(d2, K))

        # initialize mu
        if mu is None:
            mu = np.zeros(self.I)

        # initialize thetas / deltas
        tilde_theta = 0.5
        theta = 0.5
        tilde_delta = self.update_delta(
            K, tilde_theta, self.tilde_lambda0, self.tilde_lambda1, self.eta
        )
        delta = self.update_delta(K, theta, self.lambda0, self.lambda1, self.eta)

        # momentum / lag
        A_lag = A.copy()
        B_lag = B.copy()
        A_momentum = self.update_momentum(A, A_lag, 2)
        B_momentum = self.update_momentum(B, B_lag, 2)

        logLik = []

        for iter_ in range(2, max_iter):
            logLik.append(
                self.obj_function(
                    mu,
                    A,
                    B,
                    theta,
                    self.lambda0,
                    self.lambda1,
                    tilde_theta,
                    self.tilde_lambda0,
                    self.tilde_lambda1,
                )
            )

            # update A
            A_momentum = self.update_momentum(A, A_lag, iter_)
            grad_A = self.gradient("A", mu, A_momentum, B_momentum)
            tilde_Z = A_momentum - self.eta * grad_A
            A_lag = A.copy()

            for i in range(d1):
                A[i, :] = self.SSGL(
                    A_momentum[i, :],
                    tilde_theta,
                    tilde_Z[i, :],
                    tilde_delta,
                    self.eta,
                    self.tilde_lambda0,
                    self.tilde_lambda1,
                )

            # update B
            B_momentum = self.update_momentum(B, B_lag, iter_)
            grad_B = self.gradient("B", mu, A_momentum, B_momentum)
            Z = B_momentum - self.eta * grad_B
            B_lag = B.copy()

            for j in range(d2):
                B[j, :] = self.SSGL(
                    B_momentum[j, :],
                    theta,
                    Z[j, :],
                    delta,
                    self.eta,
                    self.lambda0,
                    self.lambda1,
                )

            # check if to update eta
            if (
                self.log_likelihood(
                    mu,
                    tilde_Z,
                    B_momentum,
                    theta,
                    self.lambda0,
                    self.lambda1,
                    tilde_theta,
                    self.tilde_lambda0,
                    self.tilde_lambda1,
                )
                - self.log_likelihood(
                    mu,
                    A_momentum,
                    B_momentum,
                    theta,
                    self.lambda0,
                    self.lambda1,
                    tilde_theta,
                    self.tilde_lambda0,
                    self.tilde_lambda1,
                )
                + np.sum(grad_A * (A_momentum - A))
                - np.sum(((A_momentum - A) / self.eta) ** 2)
                > 0
            ):
                self.eta = self.shrink * self.eta

            # update theta and delta every 10 iterations
            if iter_ % 10 == 0:
                tilde_theta = self.update_theta(A, self.tilde_alpha, self.tilde_beta)
                theta = self.update_theta(B, self.alpha, self.beta)
                tilde_delta = self.update_delta(
                    K, tilde_theta, self.tilde_lambda0, self.tilde_lambda1, self.eta
                )
                delta = self.update_delta(
                    K, theta, self.lambda0, self.lambda1, self.eta
                )

            # kept original behavior exactly
            # mu = self.update_mu(mu, A_momentum, B_momentum) * 0

            # convergence
            norm_A = linalg.norm(A - A_lag) / (linalg.norm(A_lag) + 1e-8)
            norm_B = linalg.norm(B - B_lag) / (linalg.norm(B_lag) + 1e-8)

            if max(norm_A, norm_B) < tol:
                break

        print(f"Finished with iterations of {iter_}")

        self.mu = mu
        self.A = A
        self.B = B

        return mu, A, B, logLik

    def summary(self):
        print("Model parameters summary:\n")
        for k, v in vars(self).items():
            if isinstance(v, np.ndarray):
                print(f"{k:13s} -> array shape={v.shape}")
            else:
                print(f"{k:13s} -> {v}")

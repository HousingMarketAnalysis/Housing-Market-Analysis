import numpy as np
import time


def DOA(X, obj_func, lb, ub, T):
    # Dollmaker Optimization Algorithm (DOA)
    N, dim = X.shape
    # Evaluate initial population
    F = np.apply_along_axis(obj_func, 1, X)

    # Best solution tracking
    best_idx = np.argmin(F)
    X_best = X[best_idx].copy()
    F_best = F[best_idx]
    Convergence = np.zeros(T)
    ct = time.time()
    # Main DOA loop
    for t in range(T):
        for i in range(N):
            # Phase 1: Pattern selection and sewing (Exploration)
            P = X_best
            r = np.random.rand()
            X_P1 = X[i] + r * (P - X[i])  # Eq (4)
            X_P1 = np.clip(X_P1, lb[i], ub[i])  # Bound check

            # Evaluate new position
            F_P1 = obj_func(X_P1)

            # Update if better
            if F_P1 < F[i]:
                X[i] = X_P1
                F[i] = F_P1

            # Phase 2: Beautifying the details of the doll (Exploitation)
            r2 = np.random.rand()
            X_P2 = X[i] + (1 - 2 * r2) * (ub[i] - lb[i])  # Eq (6)
            X_P2 = np.clip(X_P2, lb[i], ub[i])  # Bound check

            # Evaluate new position
            F_P2 = obj_func(X_P2)

            # Update if better
            if F_P2 < F[i]:
                X[i] = X_P2
                F[i] = F_P2

        # Update global best solution
        best_idx = np.argmin(F)
        if F[best_idx] < F_best:
            X_best = X[best_idx].copy()
            F_best = F[best_idx]
        Convergence[t] = F_best
    ct = time.time() - ct
    return F_best, Convergence, X_best, ct
import numpy as np
import time


def AOA(X, objfun, xmin, xmax, Max_iter):
    # Addax Optimisation Algorithm (AOA)
    # Population initialization (Eq. 1 & Eq. 2)
    Npop, dim = X.shape
    # Objective function evaluation (Eq. 3)
    Fit = objfun(X)
    best_idx = np.argmin(Fit)
    bestsol = X[best_idx].copy()
    bestfit = Fit[best_idx]
    fitness_curve = np.zeros(Max_iter)
    start_time = time.time()
    for t in range(1, Max_iter + 1):
        for i in range(Npop):
            # Phase 1: Foraging process (Exploration)
            # Candidate areas CA_i (Eq. 4)
            better_idx = np.where(Fit < Fit[i])[0]
            if len(better_idx) > 0:
                k = np.random.choice(better_idx)
                SA = X[k].copy()
            else:
                SA = bestsol.copy()
            # New position using foraging model (Eq. 5)
            X_p1 = X[i].copy()
            for j in range(dim):
                r = np.random.rand()  # r_ij ∈ [0,1]
                I = np.random.choice([1, 2])  # I_ij ∈ {1,2}
                X_p1[j] = X[i, j] + r * (SA[j] - I * X[i, j])
            X_p1 = np.clip(X_p1, xmin[i], xmax[i])
            Fit_p1 = objfun(X_p1)
            # Greedy selection (Eq. 6)
            if Fit_p1 <= Fit[i]:
                X[i] = X_p1
                Fit[i] = Fit_p1
            # Phase 2: Digging skill (Exploitation)
            # New position using digging model (Eq. 7)
            X_p2 = X[i].copy()
            for j in range(dim):
                r = np.random.rand()
                X_p2[j] = X[i, j] + (1 - 2 * r) * (
                        (xmax[i, j] - xmin[i, j]) / t
                )
            # Boundary control
            X_p2 = np.clip(X_p2, xmin[i], xmax[i])
            # Fitness evaluation
            Fit_p2 = objfun(X_p2)
            # Greedy selection (Eq. 8)
            if Fit_p2 <= Fit[i]:
                X[i] = X_p2
                Fit[i] = Fit_p2
        # Update global best solution
        best_idx = np.argmin(Fit)
        if Fit[best_idx] < bestfit:
            bestfit = Fit[best_idx]
            bestsol = X[best_idx].copy()
        fitness_curve[t - 1] = bestfit
    timecost = time.time() - start_time
    return bestfit, fitness_curve, bestsol, timecost

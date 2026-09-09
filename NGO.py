import numpy as np
import time


def NGO(X, objective, Lowerbound, Upperbound, Max_iterations):
    # Northern Goshawk Optimization (NGO)
    [Search_Agents, dimensions] = X.shape
    Lowerbound = np.ones(dimensions) * Lowerbound
    Upperbound = np.ones(dimensions) * Upperbound
    X_new = np.zeros_like(X)
    fit = np.zeros(Search_Agents)
    fit_new = np.zeros(Search_Agents)
    NGO_curve = np.zeros(Max_iterations)
    ct = time.time()
    for i in range(Search_Agents):
        L = X[i, :]
        fit[i] = objective(L)

    for t in range(Max_iterations):
        best, blocation = fit.min(), fit.argmin()

        if t == 0:
            xbest, fbest = X[blocation, :], best
        elif best < fbest:
            fbest = best
            xbest = X[blocation, :]

        for i in range(Search_Agents):
            # Phase 1: Exploration
            I = round(1 + np.random.rand())
            k = np.random.randint(0, Search_Agents)
            P = X[k, :]
            F_P = fit[k]

            if fit[i] > F_P:
                X_new[i, :] = X[i, :] + np.random.rand(dimensions) * (P - I * X[i, :])
            else:
                X_new[i, :] = X[i, :] + np.random.rand(dimensions) * (X[i, :] - P)

            X_new[i, :] = np.maximum(X_new[i, :], Lowerbound[i, :])
            X_new[i, :] = np.minimum(X_new[i, :], Upperbound[i, :])

            L = X_new[i, :]
            fit_new[i] = objective(L)

            if fit_new[i] < fit[i]:
                X[i, :] = X_new[i, :]
                fit[i] = fit_new[i]

            # Phase 2: Exploitation
            R = 0.02 * (1 - t / Max_iterations)
            X_new[i, :] = X[i, :] + (-R + 2 * R * np.random.rand(dimensions)) * X[i, :]

            X_new[i, :] = np.maximum(X_new[i, :], Lowerbound[i, :])
            X_new[i, :] = np.minimum(X_new[i, :], Upperbound[i, :])

            L = X_new[i, :]
            fit_new[i] = objective(L)

            if fit_new[i] < fit[i]:
                X[i, :] = X_new[i, :]
                fit[i] = fit_new[i]

        best_so_far = fbest  # save the best solution so far
        average = np.mean(fit)
        Score = fbest
        Best_pos = xbest
        NGO_curve[t] = Score
    ct = time.time() - ct
    best_fitness = Score
    convergence_curve = NGO_curve
    best_solution = Best_pos
    return best_fitness, convergence_curve, best_solution, ct

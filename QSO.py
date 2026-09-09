import numpy as np
import time


def QSO(Population, objective_function, lb, ub, Max_iter):
    # Quokka Swarm Optimization (QSO)
    Pop_Size, Dim = Population.shape

    # Parameter Settings
    T_range = [0.2, 0.44]
    H_range = [0.3, 0.65]
    N_range = [0, 1]

    # Step 1: Initialize
    Best_Solution = Population[0, :].copy()
    Best_Fitness = objective_function(Best_Solution)
    D_o = np.random.uniform(0, 1)
    T = T_range[0] + (T_range[1] - T_range[0]) * np.random.rand()
    H = H_range[0] + (H_range[1] - H_range[0]) * np.random.rand()
    N = N_range[0] + (N_range[1] - N_range[0]) * np.random.rand()

    Convergence = np.zeros(Max_iter)

    start_time = time.time()

    for iteration in range(Max_iter):
        Fitness_Values = np.zeros(Pop_Size)

        # Evaluate Fitness
        for i in range(Pop_Size):
            Fitness_Values[i] = objective_function(Population[i, :])

        min_fitness = np.min(Fitness_Values)
        best_idx = np.argmin(Fitness_Values)

        if min_fitness < Best_Fitness:
            Best_Solution = Population[best_idx, :].copy()
            Best_Fitness = min_fitness

        New_Population = Population.copy()

        # Update Population
        for i in range(Pop_Size):
            r = np.random.rand()
            fit_i = Fitness_Values[i]
            delta_w = np.abs(fit_i - Best_Fitness) / (fit_i + Best_Fitness + 1e-10)
            delta_X = Best_Solution - Population[i]
            D = ((T + H) / (0.8 * D_o)) + delta_w * r * delta_X

            New_Population[i, :] = (Population[i, :] - T * D +
                                    H * (Best_Solution - Population[i, :]) +
                                    N * (np.random.rand(Dim) - 0.5))
            New_Population[i, :] = (Population[i, :] - T * D +
                                    H * (Best_Solution - Population[i, :]) +
                                    N * (np.random.rand(Dim) - 0.5))

            # Apply Boundary Constraints
            New_Population[i, :] = np.clip(New_Population[i, :], lb[i, :], ub[i, :])

        Population = New_Population.copy()

        # Evaluate Fitness Again
        for i in range(Pop_Size):
            Fitness_Values[i] = objective_function(Population[i, :])

        min_fitness = np.min(Fitness_Values)
        best_idx = np.argmin(Fitness_Values)
        Best_Solution = Population[best_idx, :].copy()
        Best_Fitness = min_fitness
        Convergence[iteration] = Best_Fitness

        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}: Best Fitness = {Best_Fitness:.6f}")

    Computation_Time = time.time() - start_time
    return Best_Fitness, Convergence, Best_Solution, Computation_Time

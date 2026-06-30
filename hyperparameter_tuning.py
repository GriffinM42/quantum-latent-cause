import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader
import quantum_causal_inference as qci
import numpy as np

# Pure State (00 Bell State)
esti_state1 = np.array([[0.5, 0, 0, 0.5], 
                       [0, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [0.5, 0, 0, 0.5]])
expected_result1 = "not latent (common entropy above threshold)"
# Evenly Distributed
esti_state2 = np.array([[0.25, 0, 0, 0], 
                       [0, 0.25, 0, 0], 
                       [0, 0, 0.25, 0], 
                       [0, 0, 0, 0.25]])
expected_result2 = "not latent (too little dependence)"
# Dephased GHZ
esti_state3 = np.array([[0.5, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [0, 0, 0, 0.5]])
expected_result3 = "latent Markovizing witness"

dx = 2
dy = 2
dz = 2

problem1 = qci.QProblem(esti_state1, dx, dy, dz)
problem2 = qci.QProblem(esti_state2, dx, dy, dz)
problem3 = qci.QProblem(esti_state3, dx, dy, dz)

null_fam = [qci.QProblem(esti_state1, dx, dy, dz), qci.QProblem(esti_state2, dx, dy, dz)]


def objective(trial):
    obj = 0

    # Define hyperparameters
    smoothing = trial.suggest_float('smoothing', 0, 1)
    damping = trial.suggest_float('damping', 0, 1)
    log_reg = trial.suggest_float('log_reg', 0, 1)
    n = trial.suggest_int('n', 10, 1000)
    sig_lvl = trial.suggest_float('sig_lvl', 0, 1)

    tolerance = trial.suggest_float('tolerance', 0, 5)
    dep_gate = trial.suggest_float('dep_gate', 0, 5)
    extern_thresh = trial.suggest_float('extern_thresh', 0, 5)
    entrop_thresh = trial.suggest_float('entrop_thresh', 0, 5)
    b0 = trial.suggest_float('b0', 0, 5)
    b1 = trial.suggest_float('b1', 0, 5)
    b2 = trial.suggest_float('b2', 0, 5)
    b3 = trial.suggest_float('b3', 0, 5)
    b4 = trial.suggest_float('b4', 0, 5)
    b5 = trial.suggest_float('b5', 0, 5)
    b6 = trial.suggest_float('b6', 0, 5)
    b7 = trial.suggest_float('b7', 0, 5)
    b8 = trial.suggest_float('b8', 0, 5)
    b9 = trial.suggest_float('b9', 0, 5)

    penalties = [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9]

    result1 = qci.QInferGraph(problem3, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
                         smoothing, damping, log_reg, n, null_fam, sig_lvl)
    
    
    # if(result1[:len(expected_result1)] != expected_result1):
    #     obj += 1

    if result1.candidate_entropies is None or len(result1.candidate_entropies) < 1:
        return 1000000
    
    return result1.candidate_entropies[0][3]
    

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials = 20)
# for trial in study.get_trials():
#     print("Trial ", trial.number, ":")
#     print("Value: ", trial.value)
#     print("Params: ", trial.params)
print("Best Hyperparameters:", study.best_params)
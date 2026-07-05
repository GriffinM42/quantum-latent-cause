import sys
sys.path.insert(1, 'C:\\Users\\gevmo\\OneDrive\\Music\\Documents\\2026 - Summer\\Causal Inference\\quantum-latent-cause')

import optuna
import lib.quantum_causal_inference as qci
import lib.inference_hyperparams as ih
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

penalties = ih.penalties

smoothing = ih.smoothing
tolerance = ih.tolerance
dep_gate = ih.dep_gate
extern_thresh = ih.extern_thresh
entrop_thresh = ih.entrop_thresh

n = 100#ih.n

sig_lvl = ih.sig_lvl

problem1 = qci.QProblem(esti_state1, dx, dy, dz)
problem2 = qci.QProblem(esti_state2, dx, dy, dz)
problem3 = qci.QProblem(esti_state3, dx, dy, dz)

null_fam = [qci.QProblem(esti_state1, dx, dy, dz), qci.QProblem(esti_state2, dx, dy, dz)]


def objective(trial):

    # Define hyperparameters
    damping = trial.suggest_float('damping', 0, 1)
    log_reg = trial.suggest_float('log_reg', 0, 1)

    result1 = qci.QInferGraph(problem3, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
                         smoothing, damping, log_reg, n, null_fam, sig_lvl)
    

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
import sys
sys.path.insert(1, 'C:\\Users\\gevmo\\OneDrive\\Music\\Documents\\2026 - Summer\\Causal Inference\\quantum-latent-cause')

import optuna
import lib.quantum_causal_inference as qci
import lib.inference_hyperparams as ih
import numpy as np

A = np.zeros((64, 64))

q = 0.4
p1 = 0.1
p2 = 0.2

for x in range(0, 4):
    for y in range(0, 4):
        for z in range(0, 4):
            prod = 1
            prod *= q if (z & 2) == 2 else (1-q)
            prod *= q if (z & 1) == 2 else (1-q)
            prod *= p1 if ((z&2) != (x&2)) else (1-p1)
            prod *= p1 if ((z&1) != (x&1)) else (1-p1)
            prod *= p2 if ((z&2) != (y&2)) else (1-p2)
            prod *= p2 if ((z&1) != (y&1)) else (1-p2)
            A[x*16 + y*4 + z][x*16 + y*4 + z] = prod

prob = qci.tr_z(A, 4, 4, 4)

dx = 4
dy = 4
dz = 4

penalties = ih.penalties

smoothing = ih.smoothing
tolerance = ih.tolerance
dep_gate = ih.dep_gate
extern_thresh = ih.extern_thresh
entrop_thresh = ih.entrop_thresh

n = ih.n

sig_lvl = ih.sig_lvl

problem = qci.QProblem(prob, dx, dy, dz)

null_fam = []


def objective(trial):

    # Define hyperparameters
    damping = trial.suggest_float('damping', 0, 1)
    log_reg = trial.suggest_float('log_reg', 0, 1)

    result1 = qci.QInferGraph(problem, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
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
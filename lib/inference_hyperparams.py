import numpy as np

# General Purpose
penalties = np.arange(0.0, 1.01, 0.01)
tolerance = 0.05
entrop_thresh = 1
extern_thresh = None
dep_gate = 0.05
smoothing = 0.01
damping = 0.2
log_reg = 0.2
n = 1000

sig_lvl = 0.2

# Optimized Dephased GHZ
damping_ghz = 0.6962200606112787
log_reg_ghz = 0.007043188253465324

# Optimized Noisy Latent
damping_noisy_lat = 0.9396656310614254
log_reg_noisy_lat = 0.009754753112145118
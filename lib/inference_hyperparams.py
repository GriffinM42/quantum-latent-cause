import numpy as np

# General Purpose
penalties = np.arange(0.0, 1.01, 0.01)
tolerance = 0.05
entrop_thresh = 1
extern_thresh = None
dep_gate = 0.001
smoothing = 0.05
damping = 0.9
log_reg = 0.05
n = 1000

sig_lvl = 0.2

# Optimized Dephased GHZ
damping_ghz = 0.9942564073845965
log_reg_ghz = 0.01429298102107901

# Optimized Noisy Latent
damping_noisy_lat = 0.7256671711265638
log_reg_noisy_lat = 0.004243307073528345
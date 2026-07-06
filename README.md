# Quantum Latent Cause

A library for the inference of causal graphs for quantum systems, including experiments to test its validity.

## lib
### quantum_causal_inference.py
- A library of functions to infer the causal structure of two quantum systems based on their estimated joint state
### inference_hyperparams.py
- Suggested hyperparameters for causal inference
- Used by the experiments to define testing conditions

## level_1_experiments
Causal inference experiments using a predefined estimated joint state

## level_2_experiments
Causal inference experiments that use quantum state tomography on a circuit performed on a simulated QPU to estimate the joint state of the systems

## level_3_experiments
Causal inference experiments that use quantum state tomography on a circuit performed on a real IBM QPU to estimate the joint state of the systems
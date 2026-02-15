# Open_Project_Winter_2025_Submission

## Technical Overview

This project implements a comprehensive pipeline for Quantum State Tomography (QST), evolving from standard linear inversion techniques to machine learning-enhanced reconstruction using Classical Shadows. The primary goal is to reconstruct a quantum density matrix  given a set of measurement outcomes, enforcing physical constraints such as Hermiticity, positive semi-definiteness, and unit trace.

## Methodology

### Standard Tomography (Baseline)

* **Measurement:** Utilized Pauli-projective measurements ($X, Y, Z$ bases).
* **Reconstruction:** Implemented Linear Inversion using the generalized Born rule.
* **Validation:** Verified against analytical ground-truth states ($|0\rangle, |1\rangle, |+\rangle$) using Frobenius norm error.

### Machine Learning Approach (Classical Shadows)

* **Architecture:** A Transformer-based neural network (`ShadowReconstructor`) processes sequences of random Pauli measurements (Shadows).
* **Physical Constraints:** The model outputs a lower-triangular matrix  to construct  via Cholesky decomposition:
$$\rho = \frac{L L^\dagger}{\text{Tr}(L L^\dagger)}$$
This strictly enforces $\rho \succeq 0$ and $\text{Tr}(\rho) = 1$.
* **Training:** Supervised learning on synthetic datasets generated via Qiskit Aer, optimizing a custom loss function combining real and imaginary reconstruction errors.

## Workflow

1. **Data Generation:** Synthetic quantum states and measurement shots are generated using `src/data_gen.py`.
2. **Training:** The `ShadowReconstructor` model is trained using `src/train.py`, which minimizes reconstruction loss while monitoring Quantum Fidelity.
3. **Scalability Analysis:** `Assignment_3` benchmarking scripts assess how fidelity and runtime degrade as the qubit count  increases.

## Results  

#### Mathematical Definitions

We evaluate reconstruction quality using **Quantum Fidelity** ($F$) and **Trace Distance** ($T$):
$$F(\rho, \sigma) = \left( \text{Tr} \sqrt{ \sqrt{\rho} \sigma \sqrt{\rho} } \right)^2$$ //
$$T(\rho, \sigma) = \frac{1}{2} || \rho - \sigma ||_1 = \frac{1}{2} \text{Tr} \left[ \sqrt{(\rho - \sigma)^\dagger (\rho - \sigma)} \right]$$

#### Numerical Results

**Table 1: Baseline Linear Inversion (Single Qubit)**
| State | Frobenius Error ($||\rho_{true} - \rho_{recon}||_F$) | //
| :--- | :--- | //
| $|0\rangle$ | $0.0170$ | //
| $|1\rangle$ | $0.0300$ | //
| $|+\rangle$ | $0.0146$ | //

**Table 2: Machine Learning Reconstruction (Classical Shadows)**
*Training results after 10 Epochs on 1000 random states.*
Metric,Final Value,Convergence Behavior
Average Fidelity,98.72%,Rose from 80.17% in Epoch 1
Trace Distance,0.0770,Dropped from 0.3642 in Epoch 1
Loss (MSE),0.0030,Stabilized after Epoch 8
Inference Latency,5.63 ms,Per-sample average on CPU

| Metric | Final Value | Convergence Behavior |
| --- | --- | --- |
| **Average Fidelity** | **98.72%** | Rose from 80.17% in Epoch 1 |
| **Trace Distance** | **0.0770** | Dropped from 0.3642 in Epoch 1 |
| **Loss (MSE)** |  | Stabilized after Epoch 8 |
| **Inference Latency** |  | Per-sample average on CPU |

---

### Final Reflection

#### Scaling Limits Observed

Our scalability study (Assignment 3) revealed the "Exponential Wall" characteristic of statevector tomography:

1. Runtime Cliff ($n \approx 14$): For full statevector reconstruction, the parameter space grows as $4^n$. While operations remain instant for low qubit counts, runtime doubles with every added qubit beyond $n=12$. Matrix multiplication becomes computationally infeasible on standard hardware beyond $n=14$.
2. Fidelity Concentration: As $n$ grows, the volume of the Hilbert space explodes. A randomly initialized model has an exponentially vanishing overlap with the target state, leading to "Barren Plateaus" where gradients vanish, making optimization difficult without specific initialization strategies.

#### Future Improvements

1. **Tensor Networks:** To overcome the $4^n$ memory bottleneck, we should replace the dense Cholesky reconstruction with Matrix Product States (MPS). This would allow us to reconstruct states with low entanglement entropy for $n > 50$.
2. **Noise-Robust Loss:** The current MSE loss assumes perfect measurements (finite-shot noise only). Integrating a noise model (e.g., depolarizing channel) directly into the loss function would improve resilience against hardware errors.
3. **Attention Mechanisms:** The current Transformer averages over shadows. Implementing a deeper attention mechanism could allow the model to weight "informative" measurements higher than noisy ones, reducing the required shot count.

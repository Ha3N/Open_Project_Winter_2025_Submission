
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

import numpy as np
import torch
import os
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_density_matrix, DensityMatrix
from qiskit_aer import AerSimulator

# Configuration
NUM_QUBITS = 1           # Single qubit for initial testing
NUM_SAMPLES = 1000       # Number of random states to generate
SHADOWS_PER_STATE = 500  # Number of measurement shots per state
OUTPUT_PATH = "src/shadow_dataset.pt"

def apply_pauli_rotation(qc, pauli_idx, qubit_idx):
    """
    Rotates the basis to measure in X, Y, or Z.
    0=I (Z-basis default), 1=X, 2=Y, 3=Z
    """
    if pauli_idx == 1:   # Measure in X basis
        qc.h(qubit_idx)
    elif pauli_idx == 2: # Measure in Y basis
        qc.sdg(qubit_idx)
        qc.h(qubit_idx)
    # If 3 (Z) or 0 (I), no rotation needed (standard Z-measure)
    return qc

def generate_dataset():
    simulator = AerSimulator()
    data_X = [] # Input: List of (Pauli, Outcome)
    data_Y = [] # Target: The actual density matrix rho

    print(f"Generating {NUM_SAMPLES} quantum states with {SHADOWS_PER_STATE} shadows each...")

    for i in range(NUM_SAMPLES):
        # 1. Generate a random valid density matrix (Ground Truth)
        rho_target = random_density_matrix(2**NUM_QUBITS, seed=i)
        
        # Store flattened target for training (Real + Imag parts)
        # We store it as a complex tensor
        target_tensor = torch.tensor(rho_target.data, dtype=torch.complex64)
        data_Y.append(target_tensor)

        # 2. Simulate Classical Shadows (Measurements)
        # We select random bases for this specific state
        # 1=X, 2=Y, 3=Z
        pauli_indices = np.random.randint(1, 4, size=(SHADOWS_PER_STATE, NUM_QUBITS))
        outcomes = []

        # Optimization: We can batch this, but loops are clearer for logic
        for s in range(SHADOWS_PER_STATE):
            qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
            
            # Initialize circuit to the random state rho
            qc.set_density_matrix(rho_target)
            
            # Apply random basis rotation
            chosen_pauli = pauli_indices[s][0] # Assuming 1 qubit
            apply_pauli_rotation(qc, chosen_pauli, 0)
            
            # Measure
            qc.measure(0, 0)
            
            # Execute
            result = simulator.run(qc, shots=1).result()
            counts = result.get_counts()
            bitstring = list(counts.keys())[0]
            
            # Convert bit '0' -> +1, bit '1' -> -1 (Eigenvalues)
            outcome = 1 if bitstring == '0' else 0 # Storing as index for embedding (0 or 1)
            outcomes.append(outcome)

        # 3. Structure the input data
        # Input shape: (SHADOWS_PER_STATE, 2) -> [Pauli_Index, Outcome_Index]
        # Pauli_Index: 1,2,3. Outcome_Index: 0,1
        input_tensor = torch.tensor(list(zip(pauli_indices.flatten(), outcomes)), dtype=torch.long)
        data_X.append(input_tensor)

        if (i+1) % 100 == 0:
            print(f"Progress: {i+1}/{NUM_SAMPLES} states generated.")

    # Convert lists to tensors
    # X shape: (NUM_SAMPLES, SHADOWS_PER_STATE, 2)
    # Y shape: (NUM_SAMPLES, 2^N, 2^N)
    dataset = {
        "x": torch.stack(data_X),
        "y": torch.stack(data_Y)
    }

    # Save to disk
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(dataset, OUTPUT_PATH)
    print(f"Dataset saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_dataset()

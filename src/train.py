
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
from model import ShadowReconstructor 

# --- Configuration ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10
DATA_PATH = "src/shadow_dataset.pt"
MODEL_SAVE_PATH = "outputs/model_weights.pt"

# --- 1. Dataset Wrapper ---
class QuantumShadowDataset(Dataset):
    def __init__(self, filepath):
        data = torch.load(filepath)
        self.x = data['x']
        self.y = data['y']
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# --- 2. Metric Helpers ---
def compute_metrics(rho_pred, rho_true):
    """
    Computes both Fidelity and Trace Distance.
    """
    # Move to CPU and high precision complex128
    rho = rho_pred.detach().cpu().to(torch.complex128)
    sigma = rho_true.detach().cpu().to(torch.complex128)
    
    batch_fid = []
    batch_trace_dist = []
    
    for i in range(rho.shape[0]):
        r = rho[i]
        s = sigma[i]
        
        # --- Metric 1: Quantum Fidelity ---
        # 1. Calculate sqrt(rho)
        evals_r, evecs_r = torch.linalg.eigh(r)
        evals_r = torch.clamp(evals_r, min=0)
        # Fix: Force diagonal to be complex
        diag_r = torch.diag(torch.sqrt(evals_r)).to(torch.complex128)
        sqrt_r = evecs_r @ diag_r @ evecs_r.mH
        
        # 2. Calculate product: sqrt(rho) * sigma * sqrt(rho)
        temp = sqrt_r @ s @ sqrt_r
        
        # 3. Calculate sqrt of that product
        evals_t, evecs_t = torch.linalg.eigh(temp)
        evals_t = torch.clamp(evals_t, min=0)
        diag_t = torch.diag(torch.sqrt(evals_t)).to(torch.complex128)
        sqrt_temp = evecs_t @ diag_t @ evecs_t.mH
        
        # Trace and square
        trace_val = torch.real(torch.trace(sqrt_temp))
        fidelity = trace_val ** 2
        batch_fid.append(fidelity.item())
        
        # --- Metric 2: Trace Distance ---
        # T(rho, sigma) = 0.5 * sum(|eigenvalues(rho - sigma)|)
        # Since rho and sigma are Hermitian, their difference is Hermitian.
        # We can use eigvalsh (stable for Hermitian matrices).
        diff = r - s
        evals_diff = torch.linalg.eigvalsh(diff)
        trace_dist = 0.5 * torch.sum(torch.abs(evals_diff))
        batch_trace_dist.append(trace_dist.item())
        
    return np.mean(batch_fid), np.mean(batch_trace_dist)

# --- 3. Training Loop ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    dataset = QuantumShadowDataset(DATA_PATH)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    
    model = ShadowReconstructor(num_qubits=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() 
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            paulis = batch_x[:, :, 0]
            outcomes = batch_x[:, :, 1]
            
            optimizer.zero_grad()
            flat_params = model(paulis, outcomes)
            rho_pred = model.get_density_matrix(flat_params)
            
            loss = criterion(rho_pred.real, batch_y.real) + criterion(rho_pred.imag, batch_y.imag)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Evaluation
        model.eval()
        total_fidelity = 0
        total_trace_dist = 0
        start_time = time.time()
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                paulis = batch_x[:, :, 0]
                outcomes = batch_x[:, :, 1]
                
                flat_params = model(paulis, outcomes)
                rho_pred = model.get_density_matrix(flat_params)
                
                fid, td = compute_metrics(rho_pred, batch_y)
                total_fidelity += fid
                total_trace_dist += td
        
        avg_fidelity = total_fidelity / len(test_loader)
        avg_trace_dist = total_trace_dist / len(test_loader)
        
        # Calculate Inference Latency per sample
        # Total time / (number of batches * batch_size)
        total_samples = len(test_loader) * BATCH_SIZE
        latency_ms = ((time.time() - start_time) / total_samples) * 1000
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Fidelity: {avg_fidelity:.4f} | TraceDist: {avg_trace_dist:.4f} | Latency: {latency_ms:.2f}ms")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()

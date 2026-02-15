
import torch
import torch.nn as nn
import torch.nn.functional as F

class ShadowReconstructor(nn.Module):
    def __init__(self, num_qubits, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.d = 2 ** num_qubits  # Dimension of the Hilbert space
        self.output_dim = self.d ** 2  # Total parameters needed for L (complex)
        
        # 1. Embedding Layer
        # Inputs are (Pauli_Index, Outcome). 
        # Pauli indices: 0=I, 1=X, 2=Y, 3=Z. Outcome: 0=(-1), 1=(+1)
        self.pauli_embed = nn.Embedding(4, embed_dim)
        self.outcome_embed = nn.Embedding(2, embed_dim)
        
        # 2. Transformer Encoder (Process the set of shadows)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Prediction Head
        self.fc_hidden = nn.Linear(embed_dim, 128)
        # We output 2 * d^2 values to account for Real and Imaginary parts of L
        self.fc_output = nn.Linear(128, self.d * self.d * 2) 

    def forward(self, paulis, outcomes):
        """
        Args:
            paulis: (batch_size, num_shadows) - Integers representing Pauli basis
            outcomes: (batch_size, num_shadows) - Integers representing measurement results
        """
        # A. Embed and Combine
        x = self.pauli_embed(paulis) + self.outcome_embed(outcomes)
        
        # B. Self-Attention (Learn correlations between snapshots)
        x = self.transformer(x)
        
        # C. Global Pooling (Enforce Permutation Invariance)
        # We average over the 'num_shadows' dimension
        x = x.mean(dim=1) 
        
        # D. Project to Matrix Space
        x = F.relu(self.fc_hidden(x))
        flat_params = self.fc_output(x)
        
        return flat_params

    def get_density_matrix(self, flat_params):
        """
        Reconstructs rho strictly using the Cholesky decomposition as per assignment Part 2.
        Formula: rho = (L @ L.H) / Tr(L @ L.H)
        """
        batch_size = flat_params.shape[0]
        
        # 1. Reshape into Real and Imaginary components
        params = flat_params.view(batch_size, self.d, self.d, 2)
        L_real = params[..., 0]
        L_imag = params[..., 1]
        
        # 2. Enforce Lower Triangular Structure
        # We zero out the upper triangle to make L strictly lower triangular
        tril_mask = torch.tril(torch.ones(self.d, self.d, device=flat_params.device))
        L_real = L_real * tril_mask
        L_imag = L_imag * tril_mask
        
        # 3. Construct Complex L
        L = torch.complex(L_real, L_imag)
        
        # 4. Compute Raw Density Matrix (rho_raw = L @ L_dagger)
        # This guarantees Hermiticity and PSD
        L_dagger = L.mH # Conjugate transpose
        rho_raw = torch.matmul(L, L_dagger)
        
        # 5. Enforce Unit Trace
        trace = torch.diagonal(rho_raw, dim1=-2, dim2=-1).sum(-1)
        rho = rho_raw / trace.view(-1, 1, 1)
        
        return rho

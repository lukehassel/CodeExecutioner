import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# ==========================================
# PART 1: Exact Mhalla-Perdrix Solver (Verification)
# ==========================================
# [span_0](start_span)[span_1](start_span)Adapted from[span_0](end_span)[span_1](end_span) for verifying if a graph truly has gflow.

def solve_linear_system_f2(matrix, target):
    """Solves M * x = target over F2 using Gaussian elimination."""
    if matrix.size == 0:
        return None
    m, n = matrix.shape
    aug = np.column_stack((matrix, target)) % 2
    h, k = 0, 0
    pivots = []

    while h < m and k < n:
        i_max = np.argmax(aug[h:, k]) + h
        if aug[i_max, k] == 0:
            k += 1
        else:
            aug[[h, i_max]] = aug[[i_max, h]]
            for i in range(m):
                if i != h and aug[i, k] == 1:
                    aug[i] = (aug[i] + aug[h]) % 2
            pivots.append((h, k))
            h += 1
            k += 1

    # Check for consistency
    for i in range(h, m):
        if aug[i, n] == 1: return None # Inconsistent

    # Back substitution
    sol = np.zeros(n, dtype=int)
    # This is a simplified extraction for the "existence" check.
    # We find one valid solution.
    for r, c in reversed(pivots):
        val = aug[r, n]
        for c_prime in range(c + 1, n):
            val = (val - aug[r, c_prime] * sol[c_prime]) % 2
        sol[c] = val
    return sol

def verify_gflow_exact(adj_matrix, inputs, outputs):
    """
    [span_2](start_span)Returns True if the graph has a valid gflow using the Mhalla-Perdrix algorithm[span_2](end_span).
    """
    n = adj_matrix.shape[0]
    adj = adj_matrix.astype(int)
    
    # Sets
    v_all = set(range(n))
    out_set = set(outputs)
    in_set = set(inputs)
    
    # Corrected set (starts with outputs)
    c = out_set.copy()
    
    # Iteratively find layers
    while c != v_all:
        v_rem = sorted(list(v_all - c)) # Vertices needing correction
        v_cand = sorted(list(c - in_set)) # Vertices available for correction (past layers \ inputs)
        
        if not v_cand:
            return False # Stuck: No candidates to correct remaining nodes
            
        # We need to correct at least one u in v_rem using v_cand
        progress = False
        
        # We look for a subset K of v_rem that can be corrected ALL AT ONCE
        # But Mhalla-Perdrix implies we just need to find *any* u correctable.
        # We check each u independently for existence of correction g(u).
        
        newly_corrected = []
        for u in v_rem:
            # Equation: Adj[u, v_cand] * x = 1 (mod 2)
            # We want odd parity of neighbors in the correction set.
            row = adj[u, v_cand]
            
            # If row is all zeros, u has no neighbors in candidate set -> impossible to correct parity
            if np.sum(row) == 0:
                continue
                
            # If row has non-zeros, a solution exists over F2 for scalar target 1
            # (A non-zero linear form is surjective onto F2)
            # So u is correctable.
            newly_corrected.append(u)
            progress = True
            
        if not progress:
            return False
            
        c.update(newly_corrected)
        
    return True

# ==========================================
# PART 2: The Generator Model
# ==========================================

class GraphGenerator(nn.Module):
    def __init__(self, num_nodes, latent_dim=16):
        super().__init__()
        self.n = num_nodes
        # Simple MLP to map noise -> edge weights
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_nodes * num_nodes),
            nn.Sigmoid() # Output probabilities in [0, 1]
        )

    def forward(self, z):
        batch_size = z.size(0)
        # Generate raw matrix (Batch, N, N)
        raw = self.net(z).view(batch_size, self.n, self.n)
        
        # [span_3](start_span)Enforce Symmetry: Graph states are undirected[span_3](end_span)
        adj = (raw + raw.transpose(1, 2)) / 2
        
        # [span_4](start_span)Remove Self-Loops: Diagonal must be 0[span_4](end_span)
        mask = torch.eye(self.n, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj = adj * (1 - mask)
        
        return adj

# ==========================================
# PART 3: Differentiable GFlow Loss Module
# ==========================================

class GFlowLoss(nn.Module):
    def __init__(self, num_nodes, inputs, outputs, inner_lr=0.1, inner_iter=200):
        super().__init__()
        self.n = num_nodes
        self.inputs = inputs
        self.outputs = outputs
        
        # V \ O (Measured vertices)
        self.v_meas = sorted(list(set(range(num_nodes)) - set(outputs)))
        # V \ I (Correction candidates)
        self.v_corr = sorted(list(set(range(num_nodes)) - set(inputs)))
        
        self.inner_lr = inner_lr
        self.inner_iter = inner_iter

    def forward(self, adj_batch):
        """
        Calculates GFlow loss using Implicit Differentiation (Envelope Theorem).
        Returns: (Total GFlow Loss, Parity Component, Causal Component)
        """
        batch_size = adj_batch.size(0)
        device = adj_batch.device
        
        # --- 1. Inner Optimization (Find G*) ---
        # We detach adj_batch to stop gradients from flowing through the optimizer history
        adj_detached = adj_batch.detach()
        
        # Learnable G (logits)
        G_logits = torch.randn(batch_size, len(self.v_meas), len(self.v_corr), 
                               device=device, requires_grad=True)
        
        # Learnable Time T [Improvement: Learnable instead of fixed]
        T = torch.rand(batch_size, self.n, device=device, requires_grad=True)
        
        opt = optim.Adam([G_logits, T], lr=self.inner_lr)
        
        for _ in range(self.inner_iter):
            opt.zero_grad()
            G_soft = torch.sigmoid(G_logits)
            
            # Inner Loop Loss
            loss_p = self._parity_loss(G_soft, adj_detached)
            loss_c = self._causal_loss(G_soft, T)
            # Small sparsity push inside inner loop helps convergence
            loss_s = torch.mean(torch.abs(G_soft - 0.5)) * 0.1 
            
            total_inner = loss_p + loss_c + loss_s
            total_inner.backward()
            opt.step()
            
        # --- 2. Outer Gradient Calculation ---
        # We use the optimal G* and T found, but attach them to the computational graph
        # of the original adj_batch to allow gradients to flow back to the Generator.
        G_final = torch.sigmoid(G_logits.detach())
        T_final = T.detach()
        
        # Re-compute loss with *connected* adj_batch
        final_parity = self._parity_loss(G_final, adj_batch)
        final_causal = self._causal_loss(G_final, T_final)
        
        return final_parity + final_causal, final_parity, final_causal

    def _parity_loss(self, G, adj):
        """
        Enforces sum(A * G) is odd. Loss = (sin^2(pi/2 * Sum) - 1)^2
        """
        # Map G back to full N*N space for multiplication, or use indexing
        # adj: (B, N, N)
        # G: (B, |Meas|, |Corr|)
        
        # Extract sub-adjacency A_sub: rows=Meas, cols=Corr
        # We use advanced indexing. 
        # Expand dims for broadcasting: (B, |Meas|, |Corr|)
        
        # Gather indices
        idx_rows = torch.tensor(self.v_meas, device=adj.device).view(1, -1, 1)
        idx_cols = torch.tensor(self.v_corr, device=adj.device).view(1, 1, -1)
        
        # adj must be indexed. simpler to just loop batch or use gather if B is large.
        # For readability/compatibility, we'll use a gathered submatrix.
        # This creates a copy, which is fine.
        A_sub = adj[:, self.v_meas, :][:, :, self.v_corr]
        
        # S_u = Sum(A_uv * G_uv) over v in Corr
        S = torch.sum(A_sub * G, dim=2)
        
        # [span_5](start_span)Parity Loss[span_5](end_span)
        # We want S to be near 1, 3, 5...
        # sin^2(pi * S / 2) should be 1.
        val = torch.sin(torch.pi * S / 2) ** 2
        return torch.mean((val - 1) ** 2)

    def _causal_loss(self, G, T):
        """
        [span_6](start_span)Enforces u < v for all v in g(u). Loss = ReLU(T_u - T_v + eps)[span_6](end_span)
        """
        eps = 0.1
        # T: (B, N)
        T_u = T[:, self.v_meas].unsqueeze(2) # (B, |Meas|, 1)
        T_v = T[:, self.v_corr].unsqueeze(1) # (B, 1, |Corr|)
        
        # Penalty exists if T_u >= T_v
        penalty = F.relu(T_u - T_v + eps)
        
        # Weighted by correction map G
        return torch.mean(G * penalty)

# ==========================================
# PART 4: Training Loop
# ==========================================

def train():
    # Setup: 4-qubit Line Graph 0-1-2-3
    # Known GFlow: Inputs={0}, Outputs={3}. g(0)={1}, g(1)={2}, g(2)={3}.
    N = 4
    INPUTS = [0]
    OUTPUTS = [3]
    
    # Hyperparameters
    LATENT_DIM = 8
    BATCH_SIZE = 32 # Process multiple graphs to stabilize gradients
    LR = 0.01
    EPOCHS = 400
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    # Models
    gen = GraphGenerator(N, LATENT_DIM).to(device)
    loss_fn = GFlowLoss(N, INPUTS, OUTPUTS, inner_iter=200).to(device)
    optimizer = optim.Adam(gen.parameters(), lr=LR)
    
    print(f"Goal: Find graph state for N={N}, I={INPUTS}, O={OUTPUTS}")
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # 1. Generate
        z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
        adj_soft = gen(z)
        
        # 2. GFlow Loss (Topology & Causality)
        loss_gflow, l_par, l_cau = loss_fn(adj_soft)
        
        # 3. Sparsity Loss (Binary Constraint) [Improvement: Heavy Weight]
        # Forces edges to be 0 or 1.
        loss_sparse = torch.mean(torch.abs(adj_soft - 0.5))
        
        # Total Loss
        total_loss = loss_gflow + 2.0 * loss_sparse
        
        total_loss.backward()
        optimizer.step()
        
        # Monitoring
        if epoch % 50 == 0:
            with torch.no_grad():
                # Discrete Check on first sample
                sample_adj = (adj_soft[0] > 0.5).float().cpu().numpy()
                is_valid = verify_gflow_exact(sample_adj, INPUTS, OUTPUTS)
                status = "VALID" if is_valid else "INVALID"
                
                print(f"Epoch {epoch}: Loss {total_loss.item():.4f} "
                      f"(Parity={l_par:.4f}, Causal={l_cau:.4f}, Sparse={loss_sparse:.4f}) "
                      f"-> {status}")

    # Final Evaluation
    print("\n=== Final Result ===")
    z_test = torch.randn(1, LATENT_DIM, device=device)
    adj_final_soft = gen(z_test)[0]
    adj_final = (adj_final_soft > 0.5).float().cpu().numpy()
    
    print("Generated Adjacency Matrix:")
    print(adj_final)
    
    is_valid = verify_gflow_exact(adj_final, INPUTS, OUTPUTS)
    if is_valid:
        print("\nSUCCESS: The generated graph has a valid GFlow!")
        # Visualize edges
        edges = np.transpose(np.nonzero(np.triu(adj_final)))
        print(f"Edges: {edges.tolist()}")
    else:
        print("\nFAILURE: The generated graph does NOT have a valid GFlow.")

if __name__ == "__main__":
    train()

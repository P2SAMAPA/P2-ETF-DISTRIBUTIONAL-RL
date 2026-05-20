import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CategoricalDQN(nn.Module):
    def __init__(self, input_dim, n_actions, n_atoms, v_min, v_max, hidden_dim=128):
        super().__init__()
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta = (v_max - v_min) / (n_atoms - 1)
        self.support = torch.linspace(v_min, v_max, n_atoms)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions * n_atoms)
        )

    def forward(self, x):
        batch_size = x.size(0)
        logits = self.net(x).view(batch_size, -1, self.n_atoms)
        probs = torch.softmax(logits, dim=-1)
        return probs  # (batch, n_actions, n_atoms)

    def get_distribution(self, x):
        return self.forward(x)

    def get_cvar(self, x, alpha=0.05):
        probs = self.forward(x)  # (batch, n_actions, n_atoms)
        # For each action, compute CVaR of the distribution
        support = self.support.to(x.device)
        cvar_values = []
        for b in range(probs.size(0)):
            action_cvars = []
            for a in range(probs.size(1)):
                p = probs[b, a, :]  # (n_atoms,)
                # Sort by value (support is already sorted)
                cumsum = torch.cumsum(p, dim=0)
                idx = torch.searchsorted(cumsum, alpha)
                idx = min(idx, len(support)-1)
                # CVaR = average of tail from idx to end
                tail_probs = p[idx:] / p[idx:].sum()
                cvar = (tail_probs * support[idx:]).sum()
                action_cvars.append(cvar)
            cvar_values.append(action_cvars)
        return torch.tensor(cvar_values, device=x.device)  # (batch, n_actions)

def train_c51(train_X, train_y, input_dim, n_atoms=51, v_min=-0.05, v_max=0.05,
              hidden_dim=128, lr=1e-3, epochs=50, batch_size=32, device='cpu'):
    n_actions = 1  # we treat each ETF as separate; for each model we have one action (hold/sell?) Actually we predict a distribution of next return.
    # We'll treat it as regression to categorical distribution over returns.
    # So we set n_actions = 1 (only one action: "hold"). The distribution is over next returns.
    model = CategoricalDQN(input_dim, n_actions=1, n_atoms=n_atoms,
                           v_min=v_min, v_max=v_max, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_t = torch.tensor(train_X, dtype=torch.float32).to(device)
    y_t = torch.tensor(train_y, dtype=torch.float32).to(device)
    n = len(X_t)
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            Xb = X_t[batch_idx]
            yb = y_t[batch_idx]
            # Compute target distribution
            probs = model(Xb).squeeze(1)  # (batch, n_atoms)
            # Build target distribution: one-hot at the atom index closest to yb
            target_idx = ((yb - v_min) / ((v_max - v_min) / (n_atoms - 1))).round().long()
            target_idx = torch.clamp(target_idx, 0, n_atoms-1)
            target = torch.zeros_like(probs)
            target.scatter_(1, target_idx.unsqueeze(1), 1.0)
            # Cross-entropy loss
            loss = -(target * torch.log(probs + 1e-8)).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, loss: {total_loss/len(indices):.6f}")
    return model

def predict_cvar(model, X, alpha=0.05):
    device = next(model.parameters()).device
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        cvar = model.get_cvar(X_t, alpha)
    return cvar.cpu().numpy().squeeze()

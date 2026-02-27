import torch
import torch.optim as optim
from typing import List, Dict, Any

from prism.efe import compute_efe, aif_dpo_loss, compute_information_gain, compute_pragmatic_value

class MockAPIEnv:
    def __init__(self):
        self.undocumented_endpoint = "/api/v2/secret"
    
    def call(self, url: str) -> str:
        if url == self.undocumented_endpoint:
            return "200 OK: Secret data"
        return "404 Not Found"

class Stage2APIModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.state_params = torch.nn.Parameter(torch.randn(5))
        
    def generate_api_calls(self, history: List[str], k: int) -> List[str]:
        calls = [f"/api/v1/auth", f"/api/v2/user", f"/api/v2/secret", f"/api/v1/ping"]
        return calls[:k]
        
    def predict_api_behavior(self, history: List[str]) -> torch.Tensor:
        """Predicts the probability distribution over possible API features/specs."""
        return self.state_params.unsqueeze(0)

def run_stage2_api_experiment(epochs: int = 5, k: int = 4):
    """
    Experiment 2b: API Exploration
    Tests hidden structure discovery (undocumented API behaviors).
    """
    model = Stage2APIModel()
    ref_model = Stage2APIModel()
    ref_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    env = MockAPIEnv()
    
    # Target pragmatic distribution (a complete valid spec)
    target_spec = torch.ones(1, 5) / 5.0
    
    history = []
    
    for t in range(epochs):
        optimizer.zero_grad()
        print(f"\\n--- Epoch {t+1} (API Exploration) ---")
        
        candidates = model.generate_api_calls(history, k)
        scored_candidates = []
        prior_logits = model.predict_api_behavior(history)
        
        for a_k in candidates:
            o_k = env.call(a_k)
            hist_next = history + [a_k, o_k]
            posterior_logits = model.predict_api_behavior(hist_next)
            
            ig = compute_information_gain(prior_logits, posterior_logits)
            prag = compute_pragmatic_value(posterior_logits, target_spec)
            efe = compute_efe(ig, prag)
            
            scored_candidates.append((a_k, efe.item(), o_k))
            
        scored_candidates.sort(key=lambda x: x[1])
        y_w = scored_candidates[0]
        y_l = scored_candidates[-1]
        
        print(f"Selected API Call: {y_w[0]} (EFE: {y_w[1]:.4f}) -> {y_w[2]}")
        
        # MOCK DPO Logits
        pol_chosen = model.state_params.sum().view(1) + torch.randn(1)
        pol_rejected = model.state_params.sum().view(1) + torch.randn(1)
        
        loss, _, _ = aif_dpo_loss(pol_chosen, pol_rejected, pol_chosen.detach(), pol_rejected.detach())
        loss.backward()
        optimizer.step()
        
        print(f"Model Update Loss: {loss.item():.4f}")
        history.extend([y_w[0], y_w[2]])

if __name__ == "__main__":
    run_stage2_api_experiment()

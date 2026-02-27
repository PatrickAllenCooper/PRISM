import torch
import torch.optim as optim
from typing import List, Dict, Tuple
from prism.efe import compute_efe, aif_dpo_loss, compute_pragmatic_value, compute_information_gain

class MockSQLModel(torch.nn.Module):
    """
    Mock LLM handling SQL generation.
    """
    def __init__(self):
        super().__init__()
        # Mock parameters representing the model's predictive distribution
        self.theta = torch.nn.Parameter(torch.randn(10, requires_grad=True))

    def generate_candidate_queries(self, history: List[str], k: int) -> List[str]:
        return [f"SELECT * FROM table WHERE attr = {i}" for i in range(k)]

    def predict_rule_distribution(self, history: List[str]) -> torch.Tensor:
        """
        Predictive distribution over possible hidden rules.
        """
        # A real implementation would condition on history.
        return self.theta.unsqueeze(0)

class DatabaseEnv:
    """Mock database with a hidden business rule."""
    def __init__(self):
        self.hidden_rule = "price > 50 AND category = 'tech'"
    
    def execute_query(self, query: str) -> str:
        # Mock result sets simulating rule evaluation
        return "Result set matching rule"

def run_stage2_db_experiment(epochs: int = 5, k: int = 4):
    """
    Experiment 2a: Database Exploration
    """
    model = MockSQLModel()
    ref_model = MockSQLModel() # $\pi_{\mathrm{ref}}$
    ref_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    env = DatabaseEnv()
    
    # Pragamatic task prior: a target distribution representing rule correctly stated
    target_rule_distribution = torch.zeros(1, 10)
    target_rule_distribution[0, 3] = 1.0 # arbitrary index for "correct rule"
    
    history = []
    
    for t in range(epochs):
        optimizer.zero_grad()
        print(f"\\n--- Epoch {t+1} ---")
        
        # 1. Generate K candidates
        candidates = model.generate_candidate_queries(history, k)
        
        scored_candidates = []
        prior_logits = model.predict_rule_distribution(history)
        
        # 2-4. Simulate and Evaluate candidates
        for a_k in candidates:
            o_k = env.execute_query(a_k)
            hist_next = history + [a_k, o_k]
            posterior_logits = model.predict_rule_distribution(hist_next)
            
            info_gain = compute_information_gain(prior_logits, posterior_logits)
            pragmatic_val = compute_pragmatic_value(posterior_logits, target_rule_distribution)
            
            efe = compute_efe(info_gain, pragmatic_val)
            scored_candidates.append((a_k, efe.item(), o_k))
            
        # 5. Rank and Select Preference Pairs
        scored_candidates.sort(key=lambda x: x[1]) # Ascending EFE
        y_w = scored_candidates[0] # Min EFE -> best action
        y_l = scored_candidates[-1] # Max EFE -> worst action
        
        print(f"Best query: {y_w[0]} (EFE={y_w[1]:.4f})")
        print(f"Worst query: {y_l[0]} (EFE={y_l[1]:.4f})")
        
        # 6. DPO Update (Levels 1 VFE)
        # We need mock logprobs of these actions under policy and reference
        pol_chosen = torch.randn(1, requires_grad=True) * model.theta.sum() if hasattr(model, 'theta') else torch.randn(1, requires_grad=True)
        pol_rejected = torch.randn(1, requires_grad=True) * model.theta.sum() if hasattr(model, 'theta') else torch.randn(1, requires_grad=True)
        ref_chosen = pol_chosen.detach()
        ref_rejected = pol_rejected.detach()
        
        loss, c_reward, r_reward = aif_dpo_loss(
            policy_chosen_logps=pol_chosen,
            policy_rejected_logps=pol_rejected,
            reference_chosen_logps=ref_chosen,
            reference_rejected_logps=ref_rejected
        )
        
        loss.backward()
        optimizer.step()
        print(f"AIF-DPO Loss: {loss.item():.4f}")
        
        # Extend history
        history.extend([y_w[0], y_w[2]])

if __name__ == "__main__":
    print("Running Stage 2a DB Experiment...")
    run_stage2_db_experiment()

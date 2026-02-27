import torch
import torch.optim as optim
from typing import List
from prism.efe import compute_efe, aif_dpo_loss, compute_information_gain, compute_pragmatic_value

class ComplexWebEnv:
    """Mock environment representing the World Wide Web for research."""
    def search(self, query: str) -> str:
        # Returns a mock webpage snippet
        return f"Snippet containing information about {query.split()[-1]}"

class MockSearchAgent(torch.nn.Module):
    """LLM agent simulating multi-step web research."""
    def __init__(self, vocab_size: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = torch.nn.Linear(10, vocab_size)
        self.state = torch.nn.Parameter(torch.randn(10))

    def generate_queries(self, history: List[str], k: int) -> List[str]:
        return [f"research complex topic query_{i}" for i in range(k)]

    def predict_answer_distribution(self, history: List[str]) -> torch.Tensor:
        """p_theta(answer | q, x_t)"""
        # simplified mapping
        logits = self.encoder(self.state)
        return logits.unsqueeze(0)

def run_stage3_web_experiment(questions: int = 2, budget: int = 3, k: int = 5):
    """
    Experiment 3: Multi-Step Web Search
    Demonstrates approximate EFE via noisy sampling over sequential search steps.
    """
    model = MockSearchAgent()
    ref_model = MockSearchAgent()
    ref_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    env = ComplexWebEnv()
    
    # Ground truth / preferred task answers
    target_answer_distribution = torch.zeros(1, model.vocab_size)
    target_answer_distribution[0, 42] = 1.0 

    for q_idx in range(questions):
        print(f"\\n=== Topic {q_idx+1}: Complex Synthesis Question ===")
        history = []
        
        for step in range(budget):
            optimizer.zero_grad()
            
            candidates = model.generate_queries(history, k)
            scored = []
            
            # 1. Pre-observation beliefs
            prior_logits = model.predict_answer_distribution(history)
            
            for a_k in candidates:
                # 2. Sample noisy observation
                o_k = env.search(a_k)
                hist_next = history + [a_k, o_k]
                
                # 3. Post-observation beliefs
                posterior_logits = model.predict_answer_distribution(hist_next)
                
                # 4. Estimate Information Gain (Eq 11) - noisy sample kl divergence
                ig = compute_information_gain(prior_logits, posterior_logits)
                
                # 5. Estimate Pragmatic Value
                prag = compute_pragmatic_value(posterior_logits, target_answer_distribution)
                
                efe = compute_efe(ig, prag)
                scored.append((a_k, efe.item(), o_k))
                
            scored.sort(key=lambda x: x[1])
            y_w = scored[0]
            y_l = scored[-1]
            
            print(f" Step {step+1}: Chose '{y_w[0]}' (EFE: {y_w[1]:.4f})")
            
            # DPO Update
            pol_chosen_logp = model.state.sum() * torch.randn(1)
            pol_rejected_logp = model.state.sum() * torch.randn(1)
            loss, _, _ = aif_dpo_loss(pol_chosen_logp.unsqueeze(0), pol_rejected_logp.unsqueeze(0),
                                      pol_chosen_logp.detach().unsqueeze(0), pol_rejected_logp.detach().unsqueeze(0))
            loss.backward()
            optimizer.step()
            
            print(f" AIF-DPO Loss = {loss.item():.4f}")
            history.extend([y_w[0], y_w[2]])

if __name__ == "__main__":
    run_stage3_web_experiment()

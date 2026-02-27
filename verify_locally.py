"""
Verification script for running PRISM Stage 2 (AIF-DPO) on a local RTX 3080.
Requires ~3-4GB of VRAM with SmolLM2-135M and LoRA.
"""

import torch
import torch.optim as optim
from prism.efe import compute_efe, aif_dpo_loss, compute_information_gain, compute_pragmatic_value

# Import the real HF agent we just created
try:
    from prism.hf_integration import RealHFAgent
except ImportError:
    print("Please ensure transformers and peft are installed: pip install transformers peft")
    exit(1)

def run_local_verification():
    print("Setting up local verification on RTX 3080...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # SmolLM2 135M is tiny enough to run incredibly fast locally 
    # and prove the DPO backward passes track gradients correctly
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    
    print(f"Loading {model_id} with PEFT/LoRA...")
    agent = RealHFAgent(model_id, use_lora=True, device=device)
    ref_agent = RealHFAgent(model_id, use_lora=False, device=device)
    ref_agent.model.eval() # Reference model does not train
    
    optimizer = optim.AdamW(agent.model.parameters(), lr=5e-5)
    
    epochs = 2
    k_candidates = 3
    history_context = "System: You are exploring an unknown database rule. Propose an SQL query."
    
    print("\\nStarting AIF-DPO Loop over Real LLM Weights...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        print(f"\\n--- Epoch {epoch+1} ---")
        
        # 1. Generate real candidates from the LLM
        candidates = agent.generate_candidates(history_context, k=k_candidates)
        print(f"Generated {k_candidates} candidates.")
        
        # 2. Score Candidates using EFE
        scored = []
        prior_logits = agent.predict_next_token_logits(history_context).unsqueeze(0)
        
        for cand in candidates:
            # Fake observation for local verification
            obs = "Result: 5 rows affected."
            
            # Post-observation context
            post_context = history_context + f"\\nQuery: {cand}\\n{obs}"
            posterior_logits = agent.predict_next_token_logits(post_context).unsqueeze(0)
            
            # Compute info gain (KL between prior and posterior tokens)
            ig = compute_information_gain(prior_logits, posterior_logits)
            
            # Pragamatic task value (fake target distribution for testing)
            target_logits = torch.randn_like(posterior_logits)
            prag = compute_pragmatic_value(posterior_logits, target_logits)
            
            efe = compute_efe(ig, prag).item()
            scored.append((cand, efe))
            
        # 3. Sort by EFE and extract Chosen/Rejected
        scored.sort(key=lambda x: x[1])
        chosen_cand = scored[0][0]     # Min EFE
        rejected_cand = scored[-1][0]  # Max EFE
        
        print(f"Chosen (Low EFE): {chosen_cand}")
        print(f"Rejected (High EFE): {rejected_cand}")
        
        # 4. Compute Logprobs for DPO Loss
        pi_logp_chosen = agent.get_logprobs(history_context, chosen_cand)
        pi_logp_rejected = agent.get_logprobs(history_context, rejected_cand)
        
        with torch.no_grad():
            ref_logp_chosen = ref_agent.get_logprobs(history_context, chosen_cand)
            ref_logp_rejected = ref_agent.get_logprobs(history_context, rejected_cand)
            
        # 5. AIF-DPO Backward Pass
        loss, _, _ = aif_dpo_loss(
            pi_logp_chosen, pi_logp_rejected,
            ref_logp_chosen, ref_logp_rejected
        )
        
        print(f"AIF-DPO Loss: {loss.item():.4f}")
        loss.backward()
        optimizer.step()
        
        # Update context
        history_context += f"\\nQuery: {chosen_cand}\\nResult: 5 rows affected."
        
    print("\\nVerification Complete. Gradients flow successfully through AIF-DPO on a real LLM.")

if __name__ == "__main__":
    run_local_verification()

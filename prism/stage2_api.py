"""
Stage 2b: API Exploration (Section 5.2.2 of paper.tex)

Full AIF-DPO meta-optimization loop over a mock REST API with documented
endpoints but hidden business logic. The model must discover undocumented
validation rules, rate limits, authentication flows, and state machine
transitions through probing.

The observation space is richer than Stage 2a: structured JSON responses,
error messages with varying informativeness, and session-dependent behavior.

Protocol (Algorithm 1 from paper):
  1. Generate K candidate API calls from the current policy.
  2. Execute each call against the mock API, receive responses.
  3. Score by EFE: epistemic + pragmatic components.
  4. Construct preference pairs from the EFE ranking.
  5. DPO update using real log-probabilities from policy and frozen reference.
  6. Extend history.
"""

import torch
import torch.optim as optim
import json
import os
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from prism.hf_integration import RealHFAgent
from prism.efe import compute_information_gain, compute_efe, aif_dpo_loss, compute_pragmatic_value


class MockAPIEnvironment:
    """
    Mock REST API with hidden behaviors that the model must discover.

    Documented endpoints: /users, /products, /orders
    Hidden behaviors:
      - POST /auth/token requires {"username": "admin", "password": "secret123"}
      - GET /users requires a valid auth token in the Authorization header
      - GET /products?category=hidden returns undocumented premium products
      - POST /orders with amount > 1000 triggers a hidden approval workflow
      - Rate limit: 3 requests per endpoint per session before 429
      - DELETE /users/{id} returns 403 unless token has admin scope
    """

    FULL_SPEC = (
        "Hidden behaviors: "
        "(1) POST /auth/token with admin/secret123 returns a JWT token. "
        "(2) GET /users requires Authorization: Bearer <token> header. "
        "(3) GET /products?category=hidden reveals premium products. "
        "(4) POST /orders with amount>1000 triggers approval workflow (202 Accepted). "
        "(5) Rate limit: 3 requests per endpoint per session, then 429. "
        "(6) DELETE /users/{id} requires admin scope token, else 403."
    )

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.active_token: Optional[str] = None
        self.token_scope: str = "user"
        self.rate_counts: Dict[str, int] = {}

    def _check_rate_limit(self, endpoint: str) -> Optional[Dict]:
        base = endpoint.split("?")[0].rstrip("/")
        self.rate_counts[base] = self.rate_counts.get(base, 0) + 1
        if self.rate_counts[base] > 3:
            return {
                "status": 429,
                "body": {"error": "Too Many Requests", "retry_after": 60},
            }
        return None

    def call(self, method: str, endpoint: str, headers: Optional[Dict] = None, body: Optional[Dict] = None) -> Dict:
        headers = headers or {}
        body = body or {}

        rate_resp = self._check_rate_limit(endpoint)
        if rate_resp:
            return rate_resp

        method = method.upper()
        ep = endpoint.rstrip("/")

        if ep == "/auth/token" and method == "POST":
            if body.get("username") == "admin" and body.get("password") == "secret123":
                token = hashlib.sha256(b"admin:secret123").hexdigest()[:32]
                self.active_token = token
                self.token_scope = "admin"
                return {
                    "status": 200,
                    "body": {"token": token, "scope": "admin", "expires_in": 3600},
                }
            return {"status": 401, "body": {"error": "Invalid credentials"}}

        if ep.startswith("/users"):
            auth = headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return {"status": 401, "body": {"error": "Authentication required. Use Authorization: Bearer <token>"}}

            if method == "GET":
                return {
                    "status": 200,
                    "body": {
                        "users": [
                            {"id": 1, "name": "Alice", "role": "admin"},
                            {"id": 2, "name": "Bob", "role": "user"},
                            {"id": 3, "name": "Charlie", "role": "user"},
                        ]
                    },
                }

            if method == "DELETE":
                if self.token_scope != "admin":
                    return {"status": 403, "body": {"error": "Forbidden: admin scope required for DELETE operations"}}
                return {"status": 200, "body": {"deleted": True}}

        if ep.startswith("/products"):
            if method == "GET":
                if "category=hidden" in endpoint:
                    return {
                        "status": 200,
                        "body": {
                            "products": [
                                {"id": 100, "name": "Premium Widget", "price": 9999.99, "category": "hidden"},
                                {"id": 101, "name": "Secret Gadget", "price": 4999.99, "category": "hidden"},
                            ],
                            "note": "Premium catalog - restricted access",
                        },
                    }
                return {
                    "status": 200,
                    "body": {
                        "products": [
                            {"id": 1, "name": "Widget A", "price": 29.99, "category": "standard"},
                            {"id": 2, "name": "Widget B", "price": 49.99, "category": "standard"},
                        ]
                    },
                }

        if ep == "/orders" and method == "POST":
            amount = body.get("amount", 0)
            if amount > 1000:
                return {
                    "status": 202,
                    "body": {
                        "message": "Order requires approval",
                        "approval_id": "APR-2024-001",
                        "estimated_review": "24 hours",
                    },
                }
            return {
                "status": 201,
                "body": {"order_id": "ORD-001", "status": "confirmed", "amount": amount},
            }

        return {"status": 404, "body": {"error": f"Unknown endpoint: {method} {endpoint}"}}

    def reset(self):
        self.active_token = None
        self.token_scope = "user"
        self.rate_counts = {}


def build_api_prompt(history: List[str]) -> str:
    history_text = "\n".join(history[-8:]) if history else "No API calls made yet."
    return (
        "You are exploring a REST API to discover all its behaviors, including hidden ones.\n"
        "Known endpoints: /auth/token, /users, /products, /orders\n"
        "Methods: GET, POST, PUT, DELETE\n"
        "You can include headers and JSON body.\n\n"
        f"Previous API calls and responses:\n{history_text}\n\n"
        "Write the next API call to probe the API's hidden behaviors.\n"
        "Format: METHOD /endpoint\n"
        "API call:"
    )


def build_spec_prompt(history: List[str]) -> str:
    history_text = "\n".join(history[-8:]) if history else "No API calls made yet."
    return (
        "You are exploring a REST API.\n"
        f"Previous API calls and responses:\n{history_text}\n\n"
        "Based on the evidence, predict: what are the API's undocumented behaviors?\n"
        "The hidden behaviors are:"
    )


def parse_api_call(text: str) -> Tuple[str, str, Dict, Dict]:
    """Parse generated text into method, endpoint, headers, body."""
    text = text.strip().split("\n")[0]
    parts = text.split()
    method = parts[0].upper() if parts else "GET"
    endpoint = parts[1] if len(parts) > 1 else "/products"

    valid_methods = {"GET", "POST", "PUT", "DELETE", "PATCH"}
    if method not in valid_methods:
        method = "GET"
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint

    return method, endpoint, {}, {}


def run_stage2_api_experiment(
    epochs: int = 5,
    k_candidates: int = 4,
    beta: float = 0.1,
    lr: float = 5e-5,
    results_dir: str = "results",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

    print(f"Stage 2b: Loading policy model {model_id} with LoRA on {device}...")
    agent = RealHFAgent(model_id, use_lora=True, device=device)

    print(f"Stage 2b: Loading frozen reference model...")
    ref_agent = RealHFAgent(model_id, use_lora=False, device=device)
    ref_agent.model.eval()

    optimizer = optim.AdamW(
        [p for p in agent.model.parameters() if p.requires_grad], lr=lr
    )

    env = MockAPIEnvironment()
    os.makedirs(results_dir, exist_ok=True)

    history = []
    all_results = []

    for t in range(epochs):
        optimizer.zero_grad()
        env.reset()
        print(f"\n--- Epoch {t+1}/{epochs} (API Exploration) ---")

        api_prompt = build_api_prompt(history)
        candidates = agent.generate_candidates(api_prompt, k=k_candidates, num_new_tokens=25)

        spec_prompt_pre = build_spec_prompt(history)
        prior_logits = agent.predict_next_token_logits(spec_prompt_pre).unsqueeze(0)

        scored_candidates = []
        for a_k in candidates:
            method, endpoint, headers, body = parse_api_call(a_k)

            if env.active_token is not None:
                headers["Authorization"] = f"Bearer {env.active_token}"

            response = env.call(method, endpoint, headers, body)
            resp_str = f"{response['status']} {json.dumps(response['body'])}"

            hist_extended = history + [
                f"Call: {method} {endpoint}",
                f"Response: {resp_str[:200]}",
            ]
            spec_prompt_post = build_spec_prompt(hist_extended)
            posterior_logits = agent.predict_next_token_logits(spec_prompt_post).unsqueeze(0)

            ig = compute_information_gain(prior_logits, posterior_logits)
            prag = compute_pragmatic_value(posterior_logits, prior_logits)
            efe = compute_efe(ig, prag)

            scored_candidates.append({
                "call": f"{method} {endpoint}",
                "response_status": response["status"],
                "response_body": str(response["body"])[:150],
                "info_gain": ig.item(),
                "pragmatic": prag.item(),
                "efe": efe.item(),
            })

        scored_candidates.sort(key=lambda x: x["efe"])
        y_w = scored_candidates[0]
        y_l = scored_candidates[-1]

        print(
            f"  Best:  '{y_w['call']}' -> {y_w['response_status']} "
            f"(EFE={y_w['efe']:.4f}, IG={y_w['info_gain']:.4f})"
        )
        print(
            f"  Worst: '{y_l['call']}' -> {y_l['response_status']} "
            f"(EFE={y_l['efe']:.4f}, IG={y_l['info_gain']:.4f})"
        )

        chosen_completion = y_w["call"]
        rejected_completion = y_l["call"]

        pi_logp_chosen = agent.get_logprobs(api_prompt, chosen_completion)
        pi_logp_rejected = agent.get_logprobs(api_prompt, rejected_completion)

        with torch.no_grad():
            ref_logp_chosen = ref_agent.get_logprobs(api_prompt, chosen_completion)
            ref_logp_rejected = ref_agent.get_logprobs(api_prompt, rejected_completion)

        loss, chosen_rewards, rejected_rewards = aif_dpo_loss(
            pi_logp_chosen, pi_logp_rejected,
            ref_logp_chosen, ref_logp_rejected,
            beta=beta,
        )

        loss.backward()
        optimizer.step()

        history.extend([
            f"Call: {y_w['call']}",
            f"Response: {y_w['response_status']} {y_w['response_body'][:100]}",
        ])

        epoch_result = {
            "epoch": t + 1,
            "loss": loss.item(),
            "chosen_reward": chosen_rewards.item(),
            "rejected_reward": rejected_rewards.item(),
            "best_call": y_w,
            "worst_call": y_l,
            "all_candidates": scored_candidates,
        }
        all_results.append(epoch_result)

        print(
            f"  AIF-DPO Loss: {loss.item():.4f} | "
            f"Chosen reward: {chosen_rewards.item():.4f} | "
            f"Rejected reward: {rejected_rewards.item():.4f}"
        )

    results_path = os.path.join(results_dir, "stage2b_api_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nStage 2b results saved to {results_path}")

    print(f"\nFull API spec was: {MockAPIEnvironment.FULL_SPEC}")
    loss_trajectory = [r["loss"] for r in all_results]
    print(f"Loss trajectory: {[f'{l:.4f}' for l in loss_trajectory]}")

    return all_results


if __name__ == "__main__":
    print("Running Stage 2b API Experiment...")
    run_stage2_api_experiment()

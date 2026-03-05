# PRISM Development Guide

## Project Overview

PRISM (Predictive Inference for Self-Directed Model Training via Active Free Energy Minimization) implements the framework described in `paper.tex`. The core idea is that active inference governs the preference optimization training loop of a foundation language model exploring a partially observable space.

## Architecture

### Core Modules

- `prism/efe.py` -- Information gain, pragmatic value, EFE computation, and AIF-DPO loss. Implements Equations 6, 8, and 14 from the paper.
- `prism/hf_integration.py` -- `RealHFAgent` class wrapping HuggingFace CausalLM with LoRA support. Provides `predict_next_token_logits`, `generate_candidates`, and `get_logprobs` for the AIF-DPO loop.

### Experiment Stages

All stages use `RealHFAgent` with `SmolLM2-135M-Instruct` and follow Algorithm 1 from the paper.

**Stage 1: Multi-Hop QA (Section 5.1)**
- File: `prism/stage1_qa.py`
- Status: Implemented with real model, frozen weights, no DPO training
- Purpose: Validates that EFE-based ranking identifies more informative search queries
- Environment: Simulated retrieval corpus with 8 multi-hop questions and pre-collected evidence passages
- Metrics: Information gain per step, answer distribution entropy trajectory
- Next steps: Integrate HotpotQA/MuSiQue benchmarks, implement baselines (relevance-ranked, diversity, random, ReAct)

**Stage 2a: Database Exploration (Section 5.2.1)**
- File: `prism/stage2_db.py`
- Status: Implemented with full AIF-DPO loop, SQLite database, real logprobs
- Purpose: Tests full meta-optimization loop in controlled environment with known ground truth
- Environment: SQLite database with customers/products/orders tables, hidden business rule (tier A + electronics + amount > 500 = 15% discount)
- Metrics: DPO loss trajectory, chosen/rejected rewards, EFE and info gain per epoch
- Next steps: Add rule complexity variations, database size variations, noise injection

**Stage 2b: API Exploration (Section 5.2.2)**
- File: `prism/stage2_api.py`
- Status: Implemented with full AIF-DPO loop, mock REST API, real logprobs
- Purpose: Tests hidden structure discovery with richer observation space (JSON, status codes, session state)
- Environment: Mock API with 6 hidden behaviors (auth flow, hidden products, rate limits, approval workflow, admin scope)
- Metrics: DPO loss trajectory, chosen/rejected rewards, endpoint discovery coverage
- Next steps: Add specification grading against ground truth, coverage/accuracy/efficiency metrics

**Stage 3: Multi-Step Web Search (Section 5.3)**
- File: `prism/stage3_web.py`
- Status: Implemented with full AIF-DPO loop, simulated web corpus, real logprobs
- Purpose: Tests framework robustness to noisy EFE estimates on realistic tasks
- Environment: 3 complex research topics (CRISPR, 2008 financial crisis, quantum cryptography) with 5 evidence passages each
- Metrics: DPO loss trajectory, information gain trajectory, answer quality vs reference
- Next steps: Add baselines (static DPO, curiosity bonus), expand topic set, add F1/ROUGE scoring

### Support Files

- `verify_locally.py` -- Quick local verification of AIF-DPO gradient flow on RTX 3080
- `scripts/submit_experiments.slurm` -- SLURM job script for CURC Alpine A100 nodes

## Infrastructure

### CURC Deployment
- Conda environment: `prism_env` (Python 3.12, PyTorch, transformers, trl, peft)
- HuggingFace cache: `/scratch/alpine/paco0228/hf_cache`
- Model: `SmolLM2-135M-Instruct` with LoRA (r=8, alpha=32, target q_proj/v_proj)
- Two model instances: policy (LoRA-adapted) + reference (frozen base)

### Key Design Decisions
1. Reference model is a separate frozen instance, not `policy.detach()` -- this is critical for meaningful DPO gradients
2. Log-probabilities computed via `get_logprobs()` on actual tokenized completions
3. Information gain measured as KL divergence between pre- and post-observation predictive distributions
4. Results saved to JSON in `results/` directory for analysis

## Development Priorities

### Immediate
- [ ] Run stages on CURC and verify non-trivial loss dynamics
- [ ] Confirm loss moves away from ln(2) = 0.6931
- [ ] Check that information gain values differ meaningfully across candidates

### Short-term
- [ ] Implement baselines for Stage 1 (relevance, diversity, random, ReAct)
- [ ] Add cross-stage analysis: plot epistemic vs pragmatic EFE components over time
- [ ] Add convergence criteria (EFE plateau, predictive convergence)

### Medium-term
- [ ] Scale to larger models (SmolLM2-360M, then 1.7B)
- [ ] Integrate real retrieval corpus (HotpotQA) for Stage 1
- [ ] Add EFE critic network (Section 4.2 of paper)
- [ ] Implement exploration-exploitation transition analysis (Section 5.4)

### Long-term
- [ ] Causal structure discovery instantiation (Section 6)
- [ ] Program synthesis instantiation (Section 6)
- [ ] Comparison with CD-RLHF and static DPO baselines

# PRISM: Predictive Inference for Self-Directed Model Training via Active Free Energy Minimization

This repository contains the working draft of the paper and the code implementations for the associated experiments. The framework models the exploration--exploitation tradeoff inside Large Language Models (LLMs) using an expected free energy (EFE) objective directly injected into the DPO (Direct Preference Optimization) loop—which we term **AIF-DPO**.

## Repository Structure

*   `paper.tex`: The working draft LaTeX source detailing the theoretical claims and experimental framework.
*   `references.bib`: LaTeX bibliography.
*   `prism/`: Contains the core framework and Python experiment stages.
    *   `efe.py`: Core utility functions for Expected Free Energy (EFE), Information Gain ($\mathcal{I}$), and the AIF-DPO loss objective.
    *   `hf_integration.py`: A native PyTorch/HuggingFace wrapper utilizing `peft` and LoRA to execute the AIF-DPO meta-optimization loop over real causal language models.
    *   `stage1_qa.py`: Experiment Stage 1 - Pure epistemic foraging validations for multi-hop question answering.
    *   `stage2_db.py`: Experiment Stage 2a - Controlled database rule discovery through the full synthetic DPO train loop.
    *   `stage2_api.py`: Experiment Stage 2b - Hidden API state discovery via active probing.
    *   `stage3_web.py`: Experiment Stage 3 - Multi-step web search demonstrating approximate info gain sampling in non-stationary spaces.
*   `verify_locally.py`: An end-to-end execution script utilizing a tiny HuggingFace model (`SmolLM2-135M-Instruct`) to empirically validate mathematical claims and objective gradient flows locally on an RTX 3080.
*   `scripts/submit_experiments.slurm`: SLURM batch execution script tailored to the CURC Alpine `aa100` cluster.
*   `requirements.txt`: Python package dependencies.

## Execution Guide

### 1. Local Validation (RTX 3080)
To verify the tractability, gradient flow, and basic EFE computation pipeline using a low-parameter model, run the local verification loop. This requires ~3-4GB of VRAM.

```bash
pip install -r requirements.txt peft accelerate
python verify_locally.py
```

### 2. High-Capacity Model Verification (CURC Alpine A100s)
To verify the behavioral claims (EFE transitioning fluidly from broad epistemic exploration to task-directed exploitation) over models capable of parsing complex DB/API abstractions, you must utilize the University of Colorado CURC Alpine resources.

1. SSH into the Alpine cluster and clone this repository into `/projects/$USER/`.
2. Schedule a SLURM job using the provided batch script to provision an A100 node:
```bash
sbatch scripts/submit_experiments.slurm
```

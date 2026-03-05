import torch
import torch.nn.functional as F
from typing import Callable, Dict, Any, Tuple

def compute_information_gain(
    prior_logits: torch.Tensor,
    posterior_logits: torch.Tensor,
) -> torch.Tensor:
    r"""
    Computes Information Gain $\mathcal{I}$ as the KL divergence between 
    the post-observation predictive distribution and the pre-observation predictive distribution.
    
    Eq (6) from the paper: \mathcal{I}(a, b_t) = D_{KL}[ p_\theta(\cdot | x_t, a, o) || p_\theta(\cdot | x_t) ]
    
    Args:
        prior_logits: Logits of shape (batch_size, vocab_size) representing $p_\theta(\cdot | x_t)$
        posterior_logits: Logits of shape (batch_size, vocab_size) representing $p_\theta(\cdot | x_t, a, o)$
        
    Returns:
        KL divergence (nats) of shape (batch_size,)
    """
    prior_probs = F.softmax(prior_logits, dim=-1)
    posterior_probs = F.softmax(posterior_logits, dim=-1)
    
    # KL(posterior || prior)
    # F.kl_div expects input in log-space and target in prob-space
    kl = F.kl_div(
        F.log_softmax(prior_logits, dim=-1),
        posterior_probs,
        reduction='none'
    ).sum(dim=-1)
    
    return kl

def compute_pragmatic_value(
    predictive_logits: torch.Tensor,
    reference_logits: torch.Tensor
) -> torch.Tensor:
    """
    Computes the pragmatic value term of the Expected Free Energy.
    
    Eq (8) from paper: D_{KL}[ p_\theta(o | a, x_t) || p(o | C) ]
    Here, reference_logits represent the task-preference prior $p(o | C)$.
    
    Args:
        predictive_logits: Model beliefs $p_\theta(o | a, x_t)$
        reference_logits: Task preferences $p(o | C)$
        
    Returns:
        KL divergence (nats) of shape (batch_size,)
    """
    pred_probs = F.softmax(predictive_logits, dim=-1)
    
    kl = F.kl_div(
        F.log_softmax(reference_logits, dim=-1),
        pred_probs,
        reduction='none'
    ).sum(dim=-1)
    
    return kl

def compute_efe(
    information_gain: torch.Tensor,
    pragmatic_value: torch.Tensor = None
) -> torch.Tensor:
    r"""
    Computes Expected Free Energy (EFE).
    
    Eq (8): EFE(a, b_t) = -\mathcal{I}(a, b_t) + D_{KL}[ p_\theta(o | a, x_t) || p(o | C) ]
    
    Args:
        information_gain: Epistemic value term
        pragmatic_value: Task-fulfillment value term. If None, computes pure epistemic foraging EFE.
        
    Returns:
        Expected Free Energy
    """
    efe = -information_gain
    if pragmatic_value is not None:
        efe = efe + pragmatic_value
    return efe

def aif_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the DPO loss, adapted for Active Inference (AIF-DPO).
    Eq (14) from the paper.
    
    Args:
        policy_chosen_logps: Log probabilities of the chosen actions under the current policy.
        policy_rejected_logps: Log probabilities of the rejected actions under the current policy.
        reference_chosen_logps: Log probabilities of the chosen actions under the reference policy.
        reference_rejected_logps: Log probabilities of the rejected actions under the reference policy.
        beta: Temperature parameter.
        
    Returns:
        loss, chosen_rewards, rejected_rewards
    """
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps
    
    logits = chosen_logratios - rejected_logratios
    
    # Loss: -log(sigmoid(beta * logits))
    loss = -F.logsigmoid(beta * logits).mean()
    
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()
    
    return loss, chosen_rewards, rejected_rewards

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple
from peft import get_peft_model, LoraConfig, TaskType

class RealHFAgent(torch.nn.Module):
    """
    Wraps a real HuggingFace Causal LM for AIF-DPO experiments.
    """
    def __init__(self, model_id: str, use_lora: bool = True, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if use_lora:
            # Standard LoRA config for DPO
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
            )
            self.model = get_peft_model(base_model, peft_config)
            self.model.print_trainable_parameters()
        else:
            self.model = base_model
            
        self.model.to(self.device)

    def predict_next_token_logits(self, context: str) -> torch.Tensor:
        """
        Returns the predictive distribution over the next token $p_\theta(o | a, x_t)$
        """
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Logits for the very last token
            next_token_logits = outputs.logits[0, -1, :]
        return next_token_logits

    def generate_candidates(self, context: str, k: int, num_new_tokens: int = 15) -> List[str]:
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        candidates = []
        for _ in range(k):
            # Using sampling with high temperature to get diverse candidates
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=num_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            generated = self.tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            candidates.append(generated.strip())
        return candidates

    def get_logprobs(self, prompt: str, completion: str) -> torch.Tensor:
        """
        Returns the sum of log probabilities of the completion tokens given the prompt.
        Required for the DPO loss computation.
        """
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        completion_tokens = self.tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        
        full_inputs = torch.cat([prompt_tokens, completion_tokens], dim=1)
        
        outputs = self.model(full_inputs)
        logits = outputs.logits[0, :-1, :] # Shifted for next-token prediction
        targets = full_inputs[0, 1:]
        
        # We only care about the logprobs of the completion tokens
        completion_start_idx = prompt_tokens.shape[1] - 1
        
        completion_logits = logits[completion_start_idx:]
        completion_targets = targets[completion_start_idx:]
        
        logprobs = torch.nn.functional.log_softmax(completion_logits, dim=-1)
        selected_logprobs = torch.gather(logprobs, 1, completion_targets.unsqueeze(1)).squeeze(1)
        
        # Return summed log probability representing \log \pi_\theta(y | x)
        return selected_logprobs.sum()

"""
Stage 3: Multi-Step Web Search (Section 5.3 of paper.tex)

Full AIF-DPO meta-optimization loop over a simulated web search environment
with complex research questions that require multi-step retrieval. Tests
whether the framework is robust to noisy EFE estimates.

At each step the model generates candidate search queries, executes them
against a simulated corpus, and uses the observations to construct
EFE-ranked preference pairs. The pragmatic term scores whether the current
evidence is sufficient to produce a good answer.

Key difference from Stage 1: here we DO train the model via DPO. The
preference pairs are reconstructed at each epoch because the belief state
changes as the model learns. This tests the non-stationary preference
dataset construction described in Algorithm 1.
"""

import torch
import torch.optim as optim
import json
import os
from typing import List, Dict, Tuple
from prism.hf_integration import RealHFAgent
from prism.efe import compute_information_gain, compute_efe, aif_dpo_loss, compute_pragmatic_value


RESEARCH_TOPICS = [
    {
        "id": "t1",
        "question": (
            "How did the development of CRISPR-Cas9 gene editing technology "
            "change the landscape of genetic disease treatment, and what are "
            "the current regulatory frameworks governing its clinical use?"
        ),
        "corpus": {
            "CRISPR discovery history": (
                "CRISPR sequences were first identified in E. coli by Yoshizumi Ishino in 1987. "
                "In 2012, Jennifer Doudna and Emmanuelle Charpentier demonstrated that CRISPR-Cas9 "
                "could be programmed to cut specific DNA sequences, earning them the 2020 Nobel Prize "
                "in Chemistry."
            ),
            "CRISPR clinical trials": (
                "The first CRISPR-based therapy, Casgevy (exagamglogene autotemcel), was approved "
                "by the FDA in December 2023 for sickle cell disease. Clinical trials are underway "
                "for beta-thalassemia, certain cancers, and hereditary blindness. As of 2024, over "
                "50 clinical trials involving CRISPR are registered globally."
            ),
            "gene editing regulation": (
                "The FDA regulates gene therapies under the biologics framework. The EMA requires "
                "long-term follow-up studies for gene-edited products. China updated its regulations "
                "in 2019 after the He Jiankui controversy. Germline editing remains prohibited in "
                "most jurisdictions under the Oviedo Convention."
            ),
            "CRISPR ethical concerns": (
                "Major ethical concerns include off-target effects, mosaicism in embryo editing, "
                "equitable access to expensive therapies (Casgevy costs approximately $2.2 million "
                "per treatment), and the potential for enhancement rather than therapy."
            ),
            "alternative gene editing": (
                "Base editing and prime editing are newer CRISPR variants that can make precise "
                "changes without double-strand breaks. TALENs and zinc finger nucleases preceded "
                "CRISPR but are less versatile. RNA editing approaches are also being developed."
            ),
        },
        "reference_answer": (
            "CRISPR-Cas9, developed by Doudna and Charpentier (2012), enabled precise DNA editing. "
            "The first approved therapy (Casgevy, 2023) treats sickle cell disease. "
            "Regulation varies: FDA uses biologics framework, EMA requires long-term follow-up, "
            "germline editing is broadly prohibited."
        ),
    },
    {
        "id": "t2",
        "question": (
            "What were the key factors that led to the 2008 financial crisis, "
            "and how did the regulatory response differ between the United States "
            "and the European Union?"
        ),
        "corpus": {
            "subprime mortgage crisis": (
                "The 2008 crisis was triggered by the collapse of the US housing bubble, fueled "
                "by subprime mortgage lending. Banks bundled risky mortgages into mortgage-backed "
                "securities (MBS) and collateralized debt obligations (CDOs), which were given "
                "inflated ratings by credit agencies."
            ),
            "Lehman Brothers collapse": (
                "Lehman Brothers filed for bankruptcy on September 15, 2008, the largest bankruptcy "
                "in US history. The decision not to bail out Lehman triggered a global panic. AIG "
                "was subsequently bailed out for $85 billion due to its exposure to credit default swaps."
            ),
            "Dodd-Frank Act": (
                "The Dodd-Frank Wall Street Reform and Consumer Protection Act was signed in July 2010. "
                "It created the Consumer Financial Protection Bureau, imposed stress testing on large "
                "banks, restricted proprietary trading (Volcker Rule), and mandated central clearing "
                "for standardized derivatives."
            ),
            "EU regulatory response": (
                "The EU established the European Banking Authority in 2011 and the Single Supervisory "
                "Mechanism in 2014. The Capital Requirements Directive IV implemented Basel III "
                "standards. The European Systemic Risk Board was created for macro-prudential "
                "oversight. The Banking Union remains incomplete as of 2024."
            ),
            "Basel III framework": (
                "Basel III increased minimum capital requirements from 2% to 4.5% CET1, added "
                "capital conservation and countercyclical buffers, introduced leverage ratio "
                "requirements, and established liquidity coverage ratio (LCR) and net stable "
                "funding ratio (NSFR) standards."
            ),
        },
        "reference_answer": (
            "Key factors: subprime lending, securitization of risky mortgages, credit rating failures, "
            "excessive leverage. The US responded with Dodd-Frank (2010), creating CFPB and imposing "
            "Volcker Rule. The EU created the Banking Union, EBA, and SSM, implementing Basel III "
            "through CRD IV."
        ),
    },
    {
        "id": "t3",
        "question": (
            "How does quantum computing threaten current cryptographic systems, "
            "and what post-quantum cryptography standards have been adopted?"
        ),
        "corpus": {
            "quantum computing threat": (
                "Shor's algorithm, published in 1994, can factor large integers in polynomial time "
                "on a quantum computer, breaking RSA and elliptic curve cryptography. Grover's "
                "algorithm provides quadratic speedup for brute-force search, weakening symmetric "
                "ciphers by effectively halving their key length."
            ),
            "current quantum hardware": (
                "As of 2024, the largest quantum computers have around 1,000 qubits (IBM Condor). "
                "Breaking RSA-2048 would require approximately 4,000 error-corrected logical qubits, "
                "which translates to millions of physical qubits with current error rates. Most "
                "experts estimate this is 10-20 years away."
            ),
            "NIST PQC standards": (
                "In August 2024, NIST finalized its first post-quantum cryptography standards: "
                "ML-KEM (Kyber) for key encapsulation, ML-DSA (Dilithium) for digital signatures, "
                "and SLH-DSA (SPHINCS+) for hash-based signatures. A fourth algorithm, FN-DSA "
                "(FALCON), is expected to be standardized in late 2024."
            ),
            "harvest now decrypt later": (
                "Intelligence agencies are suspected of storing encrypted communications now to "
                "decrypt them once quantum computers become powerful enough. This 'harvest now, "
                "decrypt later' threat means sensitive data with long-term confidentiality needs "
                "must be protected with post-quantum algorithms today."
            ),
            "migration challenges": (
                "Migrating to post-quantum cryptography requires updating protocols (TLS, SSH, VPN), "
                "replacing hardware security modules, and auditing all cryptographic dependencies. "
                "Hybrid approaches running classical and post-quantum algorithms simultaneously are "
                "recommended during the transition period."
            ),
        },
        "reference_answer": (
            "Shor's algorithm breaks RSA/ECC; Grover's weakens symmetric ciphers. "
            "NIST standardized ML-KEM, ML-DSA, and SLH-DSA in 2024. "
            "The 'harvest now, decrypt later' threat makes migration urgent despite "
            "quantum computers being 10-20 years from breaking current crypto."
        ),
    },
]


class WebSearchEnvironment:
    """
    Simulated web search with a corpus of documents organized by topic.
    Returns relevant passages based on keyword matching against the corpus.
    """

    def __init__(self, topics: List[Dict]):
        self.topics = {t["id"]: t for t in topics}

    def search(self, query: str, topic_id: str) -> str:
        topic = self.topics[topic_id]
        query_terms = set(query.lower().split())

        best_passage = None
        best_score = -1

        for doc_title, doc_text in topic["corpus"].items():
            title_terms = set(doc_title.lower().split())
            text_terms = set(doc_text.lower().split())
            score = len(query_terms & title_terms) * 5 + len(query_terms & text_terms)
            if score > best_score:
                best_score = score
                best_passage = doc_text

        return best_passage or list(topic["corpus"].values())[0]


def build_search_prompt(question: str, evidence: List[str]) -> str:
    evidence_text = "\n".join(f"- {e[:150]}" for e in evidence) if evidence else "None yet."
    return (
        f"Research question: {question}\n"
        f"Evidence gathered so far:\n{evidence_text}\n\n"
        "Generate a search query to find more information needed to answer this question.\n"
        "Search query:"
    )


def build_answer_prompt(question: str, evidence: List[str]) -> str:
    evidence_text = "\n".join(f"- {e[:150]}" for e in evidence) if evidence else "None yet."
    return (
        f"Research question: {question}\n"
        f"Evidence gathered so far:\n{evidence_text}\n\n"
        "Based on the evidence, provide a comprehensive answer.\n"
        "Answer:"
    )


def run_stage3_web_experiment(
    num_topics: int = 3,
    budget_per_topic: int = 3,
    k_candidates: int = 5,
    beta: float = 0.1,
    lr: float = 5e-5,
    results_dir: str = "results",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

    print(f"Stage 3: Loading policy model {model_id} with LoRA on {device}...")
    agent = RealHFAgent(model_id, use_lora=True, device=device)

    print(f"Stage 3: Loading frozen reference model...")
    ref_agent = RealHFAgent(model_id, use_lora=False, device=device)
    ref_agent.model.eval()

    optimizer = optim.AdamW(
        [p for p in agent.model.parameters() if p.requires_grad], lr=lr
    )

    env = WebSearchEnvironment(RESEARCH_TOPICS[:num_topics])
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    for topic_data in RESEARCH_TOPICS[:num_topics]:
        question = topic_data["question"]
        topic_id = topic_data["id"]
        ref_answer = topic_data["reference_answer"]

        print(f"\n=== Topic {topic_id}: {question[:70]}... ===")

        evidence = []
        topic_results = {
            "topic_id": topic_id,
            "question": question,
            "reference_answer": ref_answer,
            "steps": [],
        }

        ref_answer_logits = agent.predict_next_token_logits(
            f"The answer is: {ref_answer[:100]}"
        ).unsqueeze(0)

        for step in range(budget_per_topic):
            optimizer.zero_grad()

            search_prompt = build_search_prompt(question, evidence)
            candidates = agent.generate_candidates(search_prompt, k=k_candidates, num_new_tokens=20)

            answer_prompt_pre = build_answer_prompt(question, evidence)
            prior_logits = agent.predict_next_token_logits(answer_prompt_pre).unsqueeze(0)

            scored = []
            for a_k in candidates:
                obs = env.search(a_k, topic_id)
                answer_prompt_post = build_answer_prompt(question, evidence + [obs])
                posterior_logits = agent.predict_next_token_logits(answer_prompt_post).unsqueeze(0)

                ig = compute_information_gain(prior_logits, posterior_logits)
                prag = compute_pragmatic_value(posterior_logits, ref_answer_logits)
                efe = compute_efe(ig, prag)

                scored.append({
                    "query": a_k,
                    "observation": obs[:150],
                    "info_gain": ig.item(),
                    "pragmatic": prag.item(),
                    "efe": efe.item(),
                })

            scored.sort(key=lambda x: x["efe"])
            y_w = scored[0]
            y_l = scored[-1]

            print(
                f"  Step {step+1}: Best '{y_w['query'][:50]}' "
                f"(EFE={y_w['efe']:.4f}, IG={y_w['info_gain']:.4f})"
            )

            chosen_completion = y_w["query"]
            rejected_completion = y_l["query"]

            pi_logp_chosen = agent.get_logprobs(search_prompt, chosen_completion)
            pi_logp_rejected = agent.get_logprobs(search_prompt, rejected_completion)

            with torch.no_grad():
                ref_logp_chosen = ref_agent.get_logprobs(search_prompt, chosen_completion)
                ref_logp_rejected = ref_agent.get_logprobs(search_prompt, rejected_completion)

            loss, chosen_rewards, rejected_rewards = aif_dpo_loss(
                pi_logp_chosen, pi_logp_rejected,
                ref_logp_chosen, ref_logp_rejected,
                beta=beta,
            )

            loss.backward()
            optimizer.step()

            evidence.append(env.search(y_w["query"], topic_id))

            step_result = {
                "step": step + 1,
                "loss": loss.item(),
                "chosen_reward": chosen_rewards.item(),
                "rejected_reward": rejected_rewards.item(),
                "best": y_w,
                "worst": y_l,
                "all_candidates": scored,
            }
            topic_results["steps"].append(step_result)

            print(
                f"    AIF-DPO Loss: {loss.item():.4f} | "
                f"Chosen: {chosen_rewards.item():.4f} | "
                f"Rejected: {rejected_rewards.item():.4f}"
            )

        all_results.append(topic_results)

    results_path = os.path.join(results_dir, "stage3_web_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nStage 3 results saved to {results_path}")

    for r in all_results:
        losses = [s["loss"] for s in r["steps"]]
        print(f"Topic {r['topic_id']} loss trajectory: {[f'{l:.4f}' for l in losses]}")

    return all_results


if __name__ == "__main__":
    print("Running Stage 3 Web Search Experiment...")
    run_stage3_web_experiment()

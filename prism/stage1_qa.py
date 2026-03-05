"""
Stage 1: Validating the Preference Signal (Section 5.1 of paper.tex)

Pure epistemic foraging with a frozen model -- no DPO training.
Tests whether EFE-based ranking of candidate search queries identifies
more informative ones than random selection.

Protocol:
  1. For each multi-hop question, the frozen LLM generates K candidate queries.
  2. Each query is executed against a simulated retrieval corpus.
  3. Information gain is estimated as the KL divergence between the model's
     answer distribution before and after incorporating the retrieved evidence.
  4. The query with highest info gain is selected and added to history.
  5. Repeat for a fixed retrieval budget.

Metrics:
  - Information gained per query (average KL shift per step)
  - Cumulative information gain trajectory
  - Answer distribution entropy over time
"""

import torch
import json
import os
import time
from typing import List, Dict, Tuple
from prism.hf_integration import RealHFAgent
from prism.efe import compute_information_gain, compute_efe


QUESTIONS = [
    {
        "id": "q1",
        "question": "What was the population of the city where the first working programmable computer was built, according to the most recent census?",
        "passages": {
            "computing history Berlin": "The Z3, created by Konrad Zuse in 1941 in Berlin, Germany, is considered the world's first working programmable, fully automatic digital computer. It was destroyed in 1943 during an Allied bombing raid.",
            "Berlin population census": "According to the 2022 census, Berlin has a population of approximately 3.85 million people, making it the largest city in Germany and the most populous city in the European Union.",
            "ENIAC Philadelphia": "ENIAC, completed in 1945 at the University of Pennsylvania in Philadelphia, was one of the earliest electronic general-purpose digital computers, but it was not the first programmable computer.",
            "Philadelphia population": "Philadelphia had a population of 1,603,797 according to the 2020 United States Census, making it the sixth-most populous city in the US.",
            "Babbage London": "Charles Babbage designed the Analytical Engine in the 1830s in London, though it was never completed during his lifetime. It remained a theoretical design.",
        },
        "answer": "The Z3 was built in Berlin in 1941. Berlin has a population of approximately 3.85 million.",
    },
    {
        "id": "q2",
        "question": "Which Nobel Prize winner discovered the element named after the country where Marie Curie was born, and in what year was it discovered?",
        "passages": {
            "Marie Curie birthplace": "Marie Curie was born Maria Sklodowska on November 7, 1867, in Warsaw, which was then part of the Russian Empire but is the capital of Poland.",
            "polonium discovery": "Polonium was discovered in 1898 by Marie Curie and Pierre Curie. It was named after Marie Curie's homeland, Poland. The discovery was announced in a paper to the French Academy of Sciences.",
            "Marie Curie Nobel": "Marie Curie won two Nobel Prizes: the Nobel Prize in Physics in 1903 (shared with Pierre Curie and Henri Becquerel) and the Nobel Prize in Chemistry in 1911 for her discovery of radium and polonium.",
            "element naming conventions": "Chemical elements are often named after places, scientists, or properties. Examples include Francium (France), Germanium (Germany), and Americium (America).",
            "Pierre Curie biography": "Pierre Curie was a French physicist and Nobel laureate. He was a pioneer in crystallography, magnetism, and radioactivity, and died in a street accident in Paris in 1906.",
        },
        "answer": "Marie Curie, a Nobel Prize winner, discovered polonium in 1898. It was named after Poland, her birth country.",
    },
    {
        "id": "q3",
        "question": "What is the GDP per capita of the country that hosted the first modern Olympic Games, and how does it compare to the global average?",
        "passages": {
            "first modern Olympics": "The first modern Olympic Games were held in Athens, Greece, in 1896. They were organized by the International Olympic Committee, which had been founded by Pierre de Coubertin.",
            "Greece GDP per capita": "Greece's GDP per capita was approximately $20,867 in 2023 (nominal), ranking it among the developed economies of Southern Europe. This is below the EU average of about $38,000.",
            "global GDP per capita": "The global average GDP per capita in 2023 was approximately $13,000 (nominal). This figure masks enormous variation, from under $500 in the poorest nations to over $100,000 in the wealthiest.",
            "ancient Olympics": "The ancient Olympic Games were held in Olympia, Greece, from 776 BC to 393 AD. They were part of a cycle of Panhellenic Games held in honor of Zeus.",
            "Olympic host cities": "Since 1896, the Olympic Games have been hosted by cities around the world including Paris, London, Tokyo, and Los Angeles. The 2024 Summer Olympics were held in Paris.",
        },
        "answer": "Greece hosted the first modern Olympics in 1896. Its GDP per capita is approximately $20,867, which is above the global average of about $13,000.",
    },
    {
        "id": "q4",
        "question": "What programming language was used to write the operating system kernel running on the computer that first landed humans on the Moon?",
        "passages": {
            "Apollo guidance computer": "The Apollo Guidance Computer (AGC) was used aboard the Apollo 11 spacecraft that first landed humans on the Moon on July 20, 1969. It was developed at the MIT Instrumentation Laboratory.",
            "AGC software": "The AGC flight software was written in AGC assembly language, a custom assembly language designed specifically for the Apollo Guidance Computer. The software was developed by a team led by Margaret Hamilton at MIT.",
            "Margaret Hamilton": "Margaret Hamilton led the team that developed the on-board flight software for NASA's Apollo program. She coined the term 'software engineering' and her work was critical to the success of the Moon landing.",
            "Apollo 11 mission": "Apollo 11 was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Eagle on July 20, 1969.",
            "early programming languages": "In the 1960s, common programming languages included FORTRAN, COBOL, LISP, and various assembly languages. High-level languages were not yet suitable for real-time embedded systems.",
        },
        "answer": "The Apollo Guidance Computer used AGC assembly language, a custom assembly language designed for the Apollo missions.",
    },
    {
        "id": "q5",
        "question": "What is the elevation of the capital city of the country with the highest biodiversity index in the world?",
        "passages": {
            "biodiversity index rankings": "Brazil has consistently ranked as the country with the highest biodiversity in the world, containing approximately 15-20% of all biological species on Earth. It leads in plant species, freshwater fish species, and mammal species.",
            "Brasilia elevation": "Brasilia, the capital of Brazil, is located on the Brazilian Central Plateau at an elevation of approximately 1,172 meters (3,845 feet) above sea level. It was inaugurated as the capital in 1960.",
            "Colombia biodiversity": "Colombia ranks as the second most biodiverse country in the world, with the highest number of bird and orchid species. Its capital Bogota sits at 2,640 meters elevation.",
            "Amazon rainforest": "The Amazon rainforest, located primarily in Brazil, is the world's largest tropical rainforest and contains an estimated 10% of all species on Earth.",
            "Brazil geography": "Brazil is the largest country in South America, covering 8.5 million square kilometers. Its terrain ranges from the Amazon basin to the central plateau to southern highlands.",
        },
        "answer": "Brazil has the highest biodiversity. Its capital Brasilia sits at approximately 1,172 meters (3,845 feet) above sea level.",
    },
    {
        "id": "q6",
        "question": "How many years after the invention of the printing press was the country where it was invented unified into a single nation-state?",
        "passages": {
            "printing press invention": "Johannes Gutenberg invented the movable-type printing press around 1440 in Mainz, in the Holy Roman Empire (modern-day Germany). His most famous printed work was the Gutenberg Bible, completed around 1455.",
            "German unification": "The unification of Germany into a single nation-state occurred on January 18, 1871, when the German Empire was proclaimed at the Palace of Versailles following the Franco-Prussian War. Otto von Bismarck orchestrated the process.",
            "Holy Roman Empire": "The Holy Roman Empire was a multi-ethnic complex of territories in Western and Central Europe that existed from 962 to 1806. Despite its name, it was neither holy, nor Roman, nor an empire in the modern sense.",
            "Gutenberg impact": "Gutenberg's printing press revolutionized the production of books and the spread of knowledge in Europe, contributing to the Renaissance, the Reformation, and the Scientific Revolution.",
            "German states history": "Before unification, the German-speaking lands consisted of numerous independent states, including Prussia, Bavaria, Saxony, and others, loosely organized within the German Confederation after 1815.",
        },
        "answer": "The printing press was invented around 1440, and Germany was unified in 1871, approximately 431 years later.",
    },
    {
        "id": "q7",
        "question": "What is the deepest point in the ocean named after, and what was the first vessel to reach it?",
        "passages": {
            "Mariana Trench depth": "The Challenger Deep, located in the Mariana Trench in the western Pacific Ocean, is the deepest known point in Earth's oceans at approximately 10,935 meters (35,876 feet) below sea level.",
            "Challenger Deep naming": "The Challenger Deep is named after the HMS Challenger, a British Royal Navy corvette that conducted the Challenger expedition (1872-1876), the first major scientific survey of the deep ocean.",
            "Trieste dive": "The Trieste, a Swiss-designed, Italian-built deep-diving research bathyscaphe, became the first vessel to reach the bottom of the Challenger Deep on January 23, 1960. It was piloted by Jacques Piccard and Don Walsh.",
            "Mariana Trench location": "The Mariana Trench is located in the western Pacific Ocean, east of the Mariana Islands. It is approximately 2,550 kilometers long and 69 kilometers wide on average.",
            "deep sea exploration": "James Cameron made a solo dive to the Challenger Deep in 2012 in the Deepsea Challenger. Victor Vescovo reached the bottom in 2019 in the DSV Limiting Factor.",
        },
        "answer": "The Challenger Deep is named after HMS Challenger. The Trieste, piloted by Jacques Piccard and Don Walsh, was the first vessel to reach it in 1960.",
    },
    {
        "id": "q8",
        "question": "What was the primary export of the country where coffee was first cultivated, and what is that country's current population?",
        "passages": {
            "coffee origin": "Coffee was first cultivated in Ethiopia, where the coffee plant Coffea arabica is indigenous. According to legend, a goat herder named Kaldi discovered the energizing effect of coffee beans around the 9th century.",
            "Ethiopia exports": "Ethiopia's primary export is coffee, which accounts for approximately 30-35% of the country's total export earnings. Ethiopia is the largest coffee producer in Africa and the fifth largest in the world.",
            "Ethiopia population": "Ethiopia has a population of approximately 126 million people as of 2024, making it the second-most populous country in Africa after Nigeria and the 12th most populous in the world.",
            "coffee trade history": "Coffee spread from Ethiopia to Yemen in the 15th century, then to the rest of the Middle East, Persia, Turkey, and North Africa. It reached Europe in the 17th century.",
            "Ethiopian economy": "Ethiopia's economy is one of the fastest-growing in Africa. Besides coffee, key exports include oilseeds, gold, flowers, and khat. Agriculture employs about 70% of the population.",
        },
        "answer": "Coffee was first cultivated in Ethiopia. Coffee remains Ethiopia's primary export. The country has approximately 126 million people.",
    },
]


class QARetrievalEnvironment:
    """
    Simulated retrieval corpus for multi-hop QA.
    Returns evidence passages based on keyword overlap with the query.
    """

    def __init__(self, questions: List[Dict]):
        self.questions = {q["id"]: q for q in questions}

    def retrieve(self, query: str, question_id: str) -> str:
        q_data = self.questions[question_id]
        query_terms = set(query.lower().split())

        best_passage = None
        best_score = -1

        for topic, passage in q_data["passages"].items():
            topic_terms = set(topic.lower().split())
            passage_terms = set(passage.lower().split())
            score = len(query_terms & topic_terms) * 3 + len(query_terms & passage_terms)
            if score > best_score:
                best_score = score
                best_passage = passage

        return best_passage or list(q_data["passages"].values())[0]


def build_answer_prompt(question: str, evidence: List[str]) -> str:
    evidence_text = "\n".join(f"- {e}" for e in evidence) if evidence else "None yet."
    return (
        f"Question: {question}\n"
        f"Evidence gathered so far:\n{evidence_text}\n"
        f"Based on the evidence, the answer is:"
    )


def build_query_prompt(question: str, evidence: List[str]) -> str:
    evidence_text = "\n".join(f"- {e}" for e in evidence) if evidence else "None yet."
    return (
        f"Question: {question}\n"
        f"Evidence gathered so far:\n{evidence_text}\n"
        f"Generate a search query to find more information needed to answer this question.\n"
        f"Search query:"
    )


def run_stage1_experiment(
    num_questions: int = 8,
    budget_per_question: int = 3,
    k_candidates: int = 5,
    results_dir: str = "results",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

    print(f"Stage 1: Loading frozen model {model_id} on {device}...")
    agent = RealHFAgent(model_id, use_lora=False, device=device)
    agent.model.eval()

    env = QARetrievalEnvironment(QUESTIONS[:num_questions])
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    for q_data in QUESTIONS[:num_questions]:
        question = q_data["question"]
        q_id = q_data["id"]
        print(f"\n--- Question ({q_id}): {question[:80]}... ---")

        evidence = []
        question_results = {
            "question_id": q_id,
            "question": question,
            "steps": [],
        }

        for step in range(budget_per_question):
            query_prompt = build_query_prompt(question, evidence)
            candidates = agent.generate_candidates(query_prompt, k=k_candidates, num_new_tokens=20)

            answer_prompt_pre = build_answer_prompt(question, evidence)
            prior_logits = agent.predict_next_token_logits(answer_prompt_pre).unsqueeze(0)
            prior_entropy = -(
                torch.softmax(prior_logits, dim=-1) * torch.log_softmax(prior_logits, dim=-1)
            ).sum().item()

            scored = []
            for cand in candidates:
                obs = env.retrieve(cand, q_id)
                answer_prompt_post = build_answer_prompt(question, evidence + [obs])
                posterior_logits = agent.predict_next_token_logits(answer_prompt_post).unsqueeze(0)

                ig = compute_information_gain(prior_logits, posterior_logits).item()
                scored.append((cand, ig, obs))

            scored.sort(key=lambda x: x[1], reverse=True)

            best_query, best_ig, best_obs = scored[0]
            worst_query, worst_ig, _ = scored[-1]

            evidence.append(best_obs)

            post_entropy = -(
                torch.softmax(
                    agent.predict_next_token_logits(
                        build_answer_prompt(question, evidence)
                    ).unsqueeze(0),
                    dim=-1,
                )
                * torch.log_softmax(
                    agent.predict_next_token_logits(
                        build_answer_prompt(question, evidence)
                    ).unsqueeze(0),
                    dim=-1,
                )
            ).sum().item()

            step_result = {
                "step": step + 1,
                "best_query": best_query,
                "best_info_gain": best_ig,
                "worst_query": worst_query,
                "worst_info_gain": worst_ig,
                "prior_entropy": prior_entropy,
                "posterior_entropy": post_entropy,
                "all_candidates": [(c, ig) for c, ig, _ in scored],
            }
            question_results["steps"].append(step_result)

            print(
                f"  Step {step+1} | Best: '{best_query[:50]}' (IG={best_ig:.4f}) "
                f"| Worst: '{worst_query[:50]}' (IG={worst_ig:.4f}) "
                f"| Entropy: {prior_entropy:.2f} -> {post_entropy:.2f}"
            )

        all_results.append(question_results)

    results_path = os.path.join(results_dir, "stage1_qa_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nStage 1 results saved to {results_path}")

    avg_ig = sum(
        s["best_info_gain"]
        for r in all_results
        for s in r["steps"]
    ) / sum(len(r["steps"]) for r in all_results)
    print(f"Average information gain per step: {avg_ig:.4f}")

    return all_results


if __name__ == "__main__":
    print("Running Stage 1 Multi-Hop QA Experiment...")
    run_stage1_experiment()

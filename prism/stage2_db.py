"""
Stage 2a: Database Exploration (Section 5.2.1 of paper.tex)

Full AIF-DPO meta-optimization loop over a relational database with a hidden
business rule. The model generates SQL queries, executes them against a real
SQLite database, and uses the observations to construct EFE-ranked preference
pairs for DPO training.

The hidden rule is a deterministic function of several variables that the
model must discover through strategic querying. Information gain is
well-defined: each result set shifts the model's predictive distribution
about what future queries would return.

Protocol (Algorithm 1 from paper):
  1. Generate K candidate SQL queries from the current policy.
  2. Execute each query against the database, receive result sets.
  3. Score by EFE: epistemic term measures predictive shift, pragmatic term
     scores proximity to correctly stating the rule.
  4. Construct preference pairs from the EFE ranking.
  5. DPO update using real log-probabilities from policy and frozen reference.
  6. Extend history with selected query and result.
"""

import torch
import torch.optim as optim
import sqlite3
import json
import os
import random
from typing import List, Dict, Tuple
from prism.hf_integration import RealHFAgent
from prism.efe import compute_information_gain, compute_efe, aif_dpo_loss, compute_pragmatic_value


class DatabaseEnvironment:
    """
    SQLite-backed database with a hidden business rule.

    Hidden rule: "Tier A customers who order products in the 'electronics'
    category with order amount > 500 receive a 15% discount."

    The model must discover this rule through SQL queries.
    """

    HIDDEN_RULE = (
        "Tier A customers who order electronics products "
        "with order amount > 500 receive a 15% discount."
    )

    def __init__(self, seed: int = 42):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        random.seed(seed)
        self._create_schema()
        self._populate_data()

    def _create_schema(self):
        self.conn.executescript("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT,
                tier TEXT CHECK(tier IN ('A', 'B', 'C')),
                region TEXT,
                join_date TEXT
            );
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                category TEXT,
                base_price REAL
            );
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER REFERENCES customers(id),
                product_id INTEGER REFERENCES products(id),
                order_amount REAL,
                discount_pct REAL,
                order_date TEXT
            );
        """)

    def _populate_data(self):
        tiers = ["A", "B", "C"]
        regions = ["North", "South", "East", "West"]
        categories = ["electronics", "clothing", "food", "furniture"]

        customers = []
        for i in range(1, 51):
            tier = random.choice(tiers)
            region = random.choice(regions)
            year = random.randint(2018, 2023)
            customers.append((i, f"Customer_{i}", tier, region, f"{year}-01-15"))

        self.conn.executemany(
            "INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers
        )

        products = []
        for i, (name, cat, price) in enumerate(
            [
                ("Laptop", "electronics", 1200.0),
                ("Phone", "electronics", 800.0),
                ("Tablet", "electronics", 450.0),
                ("Headphones", "electronics", 150.0),
                ("T-Shirt", "clothing", 25.0),
                ("Jacket", "clothing", 120.0),
                ("Rice (10kg)", "food", 15.0),
                ("Coffee Beans", "food", 30.0),
                ("Desk", "furniture", 350.0),
                ("Chair", "furniture", 250.0),
                ("Monitor", "electronics", 600.0),
                ("Keyboard", "electronics", 80.0),
            ],
            start=1,
        ):
            products.append((i, name, cat, price))

        self.conn.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?)", products
        )

        orders = []
        order_id = 1
        for _ in range(200):
            cust_id = random.randint(1, 50)
            prod_id = random.randint(1, 12)

            cust = self.conn.execute(
                "SELECT tier FROM customers WHERE id=?", (cust_id,)
            ).fetchone()
            prod = self.conn.execute(
                "SELECT category, base_price FROM products WHERE id=?", (prod_id,)
            ).fetchone()

            amount = prod["base_price"] * random.uniform(0.8, 1.5)

            discount = 0.0
            if (
                cust["tier"] == "A"
                and prod["category"] == "electronics"
                and amount > 500
            ):
                discount = 15.0

            month = random.randint(1, 12)
            orders.append(
                (order_id, cust_id, prod_id, round(amount, 2), discount, f"2024-{month:02d}-15")
            )
            order_id += 1

        self.conn.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", orders
        )

    def execute_query(self, sql: str, max_rows: int = 20) -> str:
        try:
            cursor = self.conn.execute(sql)
            rows = cursor.fetchmany(max_rows)
            if not rows:
                return "Query returned 0 rows."

            columns = [desc[0] for desc in cursor.description]
            total = cursor.fetchall()
            total_count = len(rows) + len(total)

            lines = [" | ".join(columns)]
            lines.append("-" * len(lines[0]))
            for row in rows:
                lines.append(" | ".join(str(v) for v in row))

            if total_count > max_rows:
                lines.append(f"... ({total_count} total rows, showing first {max_rows})")
            else:
                lines.append(f"({total_count} rows)")

            return "\n".join(lines)
        except Exception as e:
            return f"SQL Error: {e}"

    def close(self):
        self.conn.close()


def build_query_gen_prompt(history: List[str]) -> str:
    history_text = "\n".join(history[-6:]) if history else "No queries executed yet."
    return (
        "You are exploring a database to discover a hidden business rule about discounts.\n"
        "The database has tables: customers (id, name, tier, region, join_date), "
        "products (id, name, category, base_price), "
        "orders (id, customer_id, product_id, order_amount, discount_pct, order_date).\n\n"
        f"Previous queries and results:\n{history_text}\n\n"
        "Write a SQL query to investigate what determines the discount.\n"
        "SQL:"
    )


def build_prediction_prompt(history: List[str]) -> str:
    history_text = "\n".join(history[-6:]) if history else "No queries executed yet."
    return (
        "You are exploring a database to discover a hidden business rule about discounts.\n"
        f"Previous queries and results:\n{history_text}\n\n"
        "Based on the evidence so far, predict: what determines whether a discount is applied?\n"
        "The discount rule is:"
    )


def run_stage2_db_experiment(
    epochs: int = 5,
    k_candidates: int = 4,
    beta: float = 0.1,
    lr: float = 5e-5,
    results_dir: str = "results",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

    print(f"Stage 2a: Loading policy model {model_id} with LoRA on {device}...")
    agent = RealHFAgent(model_id, use_lora=True, device=device)

    print(f"Stage 2a: Loading frozen reference model...")
    ref_agent = RealHFAgent(model_id, use_lora=False, device=device)
    ref_agent.model.eval()

    optimizer = optim.AdamW(
        [p for p in agent.model.parameters() if p.requires_grad], lr=lr
    )

    env = DatabaseEnvironment()
    os.makedirs(results_dir, exist_ok=True)

    history = []
    all_results = []

    for t in range(epochs):
        optimizer.zero_grad()
        print(f"\n--- Epoch {t+1}/{epochs} ---")

        query_prompt = build_query_gen_prompt(history)
        candidates = agent.generate_candidates(query_prompt, k=k_candidates, num_new_tokens=40)

        pred_prompt_pre = build_prediction_prompt(history)
        prior_logits = agent.predict_next_token_logits(pred_prompt_pre).unsqueeze(0)

        scored_candidates = []
        for a_k in candidates:
            a_k_clean = a_k.strip().split("\n")[0]
            o_k = env.execute_query(a_k_clean)

            hist_extended = history + [f"Query: {a_k_clean}", f"Result: {o_k[:200]}"]
            pred_prompt_post = build_prediction_prompt(hist_extended)
            posterior_logits = agent.predict_next_token_logits(pred_prompt_post).unsqueeze(0)

            ig = compute_information_gain(prior_logits, posterior_logits)
            prag = compute_pragmatic_value(posterior_logits, prior_logits)
            efe = compute_efe(ig, prag)

            scored_candidates.append({
                "query": a_k_clean,
                "observation": o_k[:200],
                "info_gain": ig.item(),
                "pragmatic": prag.item(),
                "efe": efe.item(),
            })

        scored_candidates.sort(key=lambda x: x["efe"])
        y_w = scored_candidates[0]
        y_l = scored_candidates[-1]

        print(f"  Best:  '{y_w['query'][:60]}' (EFE={y_w['efe']:.4f}, IG={y_w['info_gain']:.4f})")
        print(f"  Worst: '{y_l['query'][:60]}' (EFE={y_l['efe']:.4f}, IG={y_l['info_gain']:.4f})")

        chosen_completion = y_w["query"]
        rejected_completion = y_l["query"]

        pi_logp_chosen = agent.get_logprobs(query_prompt, chosen_completion)
        pi_logp_rejected = agent.get_logprobs(query_prompt, rejected_completion)

        with torch.no_grad():
            ref_logp_chosen = ref_agent.get_logprobs(query_prompt, chosen_completion)
            ref_logp_rejected = ref_agent.get_logprobs(query_prompt, rejected_completion)

        loss, chosen_rewards, rejected_rewards = aif_dpo_loss(
            pi_logp_chosen, pi_logp_rejected,
            ref_logp_chosen, ref_logp_rejected,
            beta=beta,
        )

        loss.backward()
        optimizer.step()

        history.extend([f"Query: {y_w['query']}", f"Result: {y_w['observation']}"])

        epoch_result = {
            "epoch": t + 1,
            "loss": loss.item(),
            "chosen_reward": chosen_rewards.item(),
            "rejected_reward": rejected_rewards.item(),
            "best_query": y_w,
            "worst_query": y_l,
            "all_candidates": scored_candidates,
        }
        all_results.append(epoch_result)

        print(
            f"  AIF-DPO Loss: {loss.item():.4f} | "
            f"Chosen reward: {chosen_rewards.item():.4f} | "
            f"Rejected reward: {rejected_rewards.item():.4f}"
        )

    env.close()

    results_path = os.path.join(results_dir, "stage2a_db_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nStage 2a results saved to {results_path}")

    print(f"\nHidden rule was: {DatabaseEnvironment.HIDDEN_RULE}")
    loss_trajectory = [r["loss"] for r in all_results]
    print(f"Loss trajectory: {[f'{l:.4f}' for l in loss_trajectory]}")

    return all_results


if __name__ == "__main__":
    print("Running Stage 2a DB Experiment...")
    run_stage2_db_experiment()

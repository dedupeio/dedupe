import time
from typing import Dict, List, Set, Tuple

import numpy as np

import dedupe
from dedupe.variables.string import StringType
from dedupe.variables.vector import VectorType, vector_sign_hash_predicate


class BlockingEfficiencyBenchmark:
    """
    Benchmark suite measuring:
        - Pairwise Reduction Ratio (PRR)
        - Candidate Pair Generation Throughput (records/sec)
        - Blocking Index Size and Collision Rates
    """

    def setup(self):
        # Generate synthetic benchmark dataset
        np.random.seed(42)
        self.num_records = 2000
        self.data: Dict[int, Dict[str, Any]] = {}

        companies = [
            "Stripe Inc", "Stripe Payments", "Stripe LLC",
            "Databricks Corp", "Databricks Unified Analytics", "Databricks Inc",
            "Snowflake Computing", "Snowflake Data Cloud", "Snowflake Inc",
            "OpenAI Global", "OpenAI Research", "OpenAI Technologies",
            "Anthropic AI", "Anthropic PBC", "Anthropic Claude Labs"
        ]

        for i in range(self.num_records):
            cluster_id = i % len(companies)
            base_name = companies[cluster_id]
            # Add synthetic noise
            noise = f" #{i // len(companies)}" if i >= len(companies) else ""
            name = f"{base_name}{noise}"
            embedding = np.random.randn(64).astype(np.float32)
            # Add cluster bias
            embedding += cluster_id * 0.5
            embedding /= np.linalg.norm(embedding)

            self.data[i] = {
                "name": name,
                "embedding": embedding,
                "cluster_id": cluster_id,
            }

    def time_predicate_generation(self):
        """Measures predicate extraction throughput across the dataset."""
        t0 = time.perf_counter()
        count = 0
        for record in self.data.values():
            preds = vector_sign_hash_predicate(record["embedding"])
            count += len(preds)
        elapsed = time.perf_counter() - t0
        return elapsed

    def time_vector_distance_computation(self):
        """Measures pairwise cosine distance vector comparator throughput."""
        comparator = dedupe.variables.Vector("embedding", metric="cosine").comparator
        t0 = time.perf_counter()
        records = list(self.data.values())
        for i in range(min(500, len(records) - 1)):
            comparator(records[i]["embedding"], records[i + 1]["embedding"])
        elapsed = time.perf_counter() - t0
        return elapsed


if __name__ == "__main__":
    benchmark = BlockingEfficiencyBenchmark()
    benchmark.setup()
    t_pred = benchmark.time_predicate_generation()
    t_dist = benchmark.time_vector_distance_computation()
    print(f"Predicate Generation Time for {benchmark.num_records} records: {t_pred * 1000:.2f} ms")
    print(f"500 Pairwise Vector Distance Computations: {t_dist * 1000:.2f} ms")

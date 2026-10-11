import os
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# ============================================================
# CPU CONFIG
# ============================================================

CPU_THREADS = os.cpu_count() or 16

torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(min(4, CPU_THREADS))

print(f"CPU threads: {torch.get_num_threads()}")
print(f"CPU logical processors: {CPU_THREADS}")


# ============================================================
# MODELS
# ============================================================

MODELS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "BGE-small": "BAAI/bge-small-en-v1.5",
    "E5-small": "intfloat/e5-small-v2",
}


# ============================================================
# POSITIVE PAIRS
# Same meaning -> should have HIGH similarity
# ============================================================

POSITIVE_PAIRS = [

    # Python
    (
        "How do I reverse a Python list?",
        "How can I reverse a list in Python?"
    ),
    (
        "How to check if a string is a palindrome?",
        "How can I test whether a string is a palindrome?"
    ),
    (
        "How do I sort a list in Python?",
        "What is the way to sort a Python list?"
    ),
    (
        "How can I remove duplicates from a list?",
        "How do I eliminate duplicate elements from a list?"
    ),

    # Programming
    (
        "What is object oriented programming?",
        "Explain OOP in simple terms."
    ),
    (
        "What is recursion?",
        "Can you explain recursive functions?"
    ),
    (
        "What is a hash table?",
        "Explain how hash tables work."
    ),
    (
        "What is an API?",
        "Explain what an API does."
    ),

    # SQL
    (
        "How do I delete rows in SQL?",
        "How can I remove records from a SQL table?"
    ),
    (
        "How do I find duplicate rows in SQL?",
        "How can I detect duplicate records in SQL?"
    ),

    # Computer science
    (
        "What is machine learning?",
        "What does ML mean?"
    ),
    (
        "What is a database?",
        "Explain what a database is."
    ),
    (
        "What is an operating system?",
        "Explain the purpose of an OS."
    ),
    (
        "What is TCP?",
        "Explain the TCP protocol."
    ),
    (
        "What is DNS?",
        "How does DNS work?"
    ),

    # General
    (
        "What is the capital of France?",
        "Which city is the capital of France?"
    ),
    (
        "How far is the Earth from the Sun?",
        "What is the distance between Earth and the Sun?"
    ),
    (
        "What causes earthquakes?",
        "Why do earthquakes happen?"
    ),
    (
        "What is photosynthesis?",
        "How does photosynthesis work?"
    ),
    (
        "What is gravity?",
        "Explain what gravity is."
    ),
]


# ============================================================
# NEGATIVE PAIRS
# Related wording but DIFFERENT meaning
# Should have LOW similarity
# ============================================================

NEGATIVE_PAIRS = [

    # Python
    (
        "How do I reverse a Python list?",
        "How do I sort a Python list?"
    ),
    (
        "How do I delete a file?",
        "How do I create a file?"
    ),
    (
        "How do I open a file in Python?",
        "How do I close a file in Python?"
    ),
    (
        "How do I add an element to a list?",
        "How do I remove an element from a list?"
    ),
    (
        "How do I find the maximum value?",
        "How do I find the minimum value?"
    ),

    # Programming concepts
    (
        "What is recursion?",
        "What is iteration?"
    ),
    (
        "What is a stack?",
        "What is a queue?"
    ),
    (
        "What is a compiler?",
        "What is an interpreter?"
    ),
    (
        "What is an array?",
        "What is a linked list?"
    ),
    (
        "What is a class?",
        "What is an object?"
    ),

    # SQL
    (
        "How do I delete rows in SQL?",
        "How do I update rows in SQL?"
    ),
    (
        "How do I find duplicate rows?",
        "How do I find missing rows?"
    ),
    (
        "What is a primary key?",
        "What is a foreign key?"
    ),
    (
        "What is an INNER JOIN?",
        "What is a LEFT JOIN?"
    ),

    # Networking
    (
        "What is TCP?",
        "What is UDP?"
    ),
    (
        "What is DNS?",
        "What is DHCP?"
    ),
    (
        "What is an IP address?",
        "What is a MAC address?"
    ),
    (
        "What is HTTP?",
        "What is HTTPS?"
    ),

    # AI / ML
    (
        "What is machine learning?",
        "What is deep learning?"
    ),
    (
        "What is supervised learning?",
        "What is unsupervised learning?"
    ),
    (
        "What is an embedding?",
        "What is a token?"
    ),
    (
        "What is a neural network?",
        "What is a decision tree?"
    ),

    # General knowledge
    (
        "What is the capital of France?",
        "What is the capital of Germany?"
    ),
    (
        "What is the largest planet?",
        "What is the smallest planet?"
    ),
    (
        "How far is Earth from the Sun?",
        "How far is Earth from the Moon?"
    ),
    (
        "What causes earthquakes?",
        "What causes volcanoes?"
    ),
    (
        "What is photosynthesis?",
        "What is cellular respiration?"
    ),
]


# ============================================================
# SETTINGS
# ============================================================

THRESHOLDS = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]

BATCH_SIZE = 32


# ============================================================
# HELPERS
# ============================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Vectors are normalized, so dot product = cosine similarity.
    """
    return float(np.dot(a, b))


def embed_single(model, text):
    """
    Embed one request and measure actual request latency.
    """
    start = time.perf_counter()

    vector = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    return vector.astype(np.float32), latency_ms


def calculate_metrics(
    positive_scores,
    negative_scores,
    threshold,
):
    """
    Positive = same meaning
    Negative = different meaning

    At threshold:
        similarity >= threshold -> CACHE HIT
        similarity < threshold  -> CACHE MISS
    """

    true_positive = sum(
        score >= threshold
        for score in positive_scores
    )

    false_negative = sum(
        score < threshold
        for score in positive_scores
    )

    false_positive = sum(
        score >= threshold
        for score in negative_scores
    )

    true_negative = sum(
        score < threshold
        for score in negative_scores
    )

    total = (
        true_positive
        + false_negative
        + false_positive
        + true_negative
    )

    accuracy = (
        (true_positive + true_negative) / total
        if total
        else 0
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    return {
        "tp": true_positive,
        "fn": false_negative,
        "fp": false_positive,
        "tn": true_negative,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# RUN MODEL
# ============================================================

def benchmark_model(model_name, model_path):

    print()
    print("=" * 70)
    print(f"{model_name}: {model_path}")
    print("=" * 70)

    print("Loading model...")

    model = SentenceTransformer(model_path)

    dimension = model.get_sentence_embedding_dimension()

    print(f"Dimension: {dimension}")

    # --------------------------------------------------------
    # Prepare all texts
    # --------------------------------------------------------

    all_texts = []

    for a, b in POSITIVE_PAIRS:
        all_texts.append(a)
        all_texts.append(b)

    for a, b in NEGATIVE_PAIRS:
        all_texts.append(a)
        all_texts.append(b)

    # Remove duplicates while preserving order
    all_texts = list(dict.fromkeys(all_texts))

    # --------------------------------------------------------
    # Batch embedding
    # --------------------------------------------------------

    start = time.perf_counter()

    embeddings = model.encode(
        all_texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    total_batch_ms = (time.perf_counter() - start) * 1000

    embeddings = embeddings.astype(np.float32)

    text_to_embedding = {
        text: embeddings[i]
        for i, text in enumerate(all_texts)
    }

    print(f"Texts embedded: {len(all_texts)}")
    print(f"Total embedding: {total_batch_ms:.1f} ms")
    print(
        f"Batch average: "
        f"{total_batch_ms / len(all_texts):.2f} ms/query"
    )

    # --------------------------------------------------------
    # Positive similarities
    # --------------------------------------------------------

    positive_scores = []

    for a, b in POSITIVE_PAIRS:

        score = cosine_similarity(
            text_to_embedding[a],
            text_to_embedding[b],
        )

        positive_scores.append(score)

    # --------------------------------------------------------
    # Negative similarities
    # --------------------------------------------------------

    negative_scores = []

    for a, b in NEGATIVE_PAIRS:

        score = cosine_similarity(
            text_to_embedding[a],
            text_to_embedding[b],
        )

        negative_scores.append(score)

    # --------------------------------------------------------
    # Print pair scores
    # --------------------------------------------------------

    print()
    print("POSITIVE PAIRS")
    print("-" * 70)

    for i, ((a, b), score) in enumerate(
        zip(POSITIVE_PAIRS, positive_scores),
        1,
    ):
        print(
            f"{i:02d}. {score:.4f} | "
            f"{a}  <->  {b}"
        )

    print()
    print("NEGATIVE PAIRS")
    print("-" * 70)

    for i, ((a, b), score) in enumerate(
        zip(NEGATIVE_PAIRS, negative_scores),
        1,
    ):
        print(
            f"{i:02d}. {score:.4f} | "
            f"{a}  <->  {b}"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    positive_avg = np.mean(positive_scores)
    positive_min = np.min(positive_scores)
    positive_max = np.max(positive_scores)

    negative_avg = np.mean(negative_scores)
    negative_min = np.min(negative_scores)
    negative_max = np.max(negative_scores)

    separation = positive_avg - negative_avg

    print()
    print("SIMILARITY STATISTICS")
    print("-" * 70)

    print(
        f"Positive avg : {positive_avg:.4f}"
    )
    print(
        f"Positive min : {positive_min:.4f}"
    )
    print(
        f"Positive max : {positive_max:.4f}"
    )

    print(
        f"Negative avg : {negative_avg:.4f}"
    )
    print(
        f"Negative min : {negative_min:.4f}"
    )
    print(
        f"Negative max : {negative_max:.4f}"
    )

    print(
        f"Separation   : {separation:.4f}"
    )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    threshold_results = []

    print()
    print("THRESHOLD ANALYSIS")
    print("-" * 70)

    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'FP':<8}"
        f"{'FN':<8}"
    )

    print("-" * 70)

    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            positive_scores,
            negative_scores,
            threshold,
        )

        threshold_results.append(
            (threshold, metrics)
        )

        print(
            f"{threshold:<12.2f}"
            f"{metrics['accuracy'] * 100:<12.1f}"
            f"{metrics['precision'] * 100:<12.1f}"
            f"{metrics['recall'] * 100:<12.1f}"
            f"{metrics['f1'] * 100:<12.1f}"
            f"{metrics['fp']:<8}"
            f"{metrics['fn']:<8}"
        )

    # --------------------------------------------------------
    # Best F1 threshold
    # --------------------------------------------------------

    best_threshold, best_metrics = max(
        threshold_results,
        key=lambda x: x[1]["f1"],
    )

    # --------------------------------------------------------
    # Actual single-request latency
    # --------------------------------------------------------

    latency_samples = []

    # Use representative requests from the positive set
    request_texts = [
        pair[0]
        for pair in POSITIVE_PAIRS
    ]

    print()
    print("REQUEST EMBEDDING LATENCY")
    print("-" * 70)

    for text in request_texts:

        _, latency_ms = embed_single(
            model,
            text,
        )

        latency_samples.append(latency_ms)

    latency_samples = np.array(latency_samples)

    avg_latency = np.mean(latency_samples)
    p50_latency = np.percentile(latency_samples, 50)
    p95_latency = np.percentile(latency_samples, 95)

    print(
        f"Average: {avg_latency:.2f} ms"
    )

    print(
        f"P50:     {p50_latency:.2f} ms"
    )

    print(
        f"P95:     {p95_latency:.2f} ms"
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "model": model_name,
        "dimension": dimension,
        "batch_ms": total_batch_ms,
        "batch_ms_per_query": (
            total_batch_ms / len(all_texts)
        ),
        "request_ms": avg_latency,
        "p50_ms": p50_latency,
        "p95_ms": p95_latency,
        "positive_avg": positive_avg,
        "positive_min": positive_min,
        "negative_avg": negative_avg,
        "negative_max": negative_max,
        "separation": separation,
        "best_threshold": best_threshold,
        "best_f1": best_metrics["f1"],
        "best_accuracy": best_metrics["accuracy"],
        "best_precision": best_metrics["precision"],
        "best_recall": best_metrics["recall"],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    results = []

    for model_name, model_path in MODELS.items():

        result = benchmark_model(
            model_name,
            model_path,
        )

        results.append(result)

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print()
    print()
    print("=" * 115)
    print("FINAL COMPARISON")
    print("=" * 115)

    print(
        f"{'Model':<12}"
        f"{'Dim':<8}"
        f"{'Req ms':<10}"
        f"{'P95 ms':<10}"
        f"{'Pos Avg':<10}"
        f"{'Neg Avg':<10}"
        f"{'Sep':<10}"
        f"{'Threshold':<12}"
        f"{'F1':<10}"
    )

    print("-" * 115)

    for r in results:

        print(
            f"{r['model']:<12}"
            f"{r['dimension']:<8}"
            f"{r['request_ms']:<10.1f}"
            f"{r['p95_ms']:<10.1f}"
            f"{r['positive_avg']:<10.4f}"
            f"{r['negative_avg']:<10.4f}"
            f"{r['separation']:<10.4f}"
            f"{r['best_threshold']:<12.2f}"
            f"{r['best_f1'] * 100:<10.1f}"
        )

    print("=" * 115)


if __name__ == "__main__":
    main()
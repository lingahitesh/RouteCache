# scripts/test_embeddings_faiss.py
"""
Step 6 definition of done:
  - Embed and index 100 queries
  - Search for 10 known near-duplicates
  - Manually confirm the top result is sensible
  - Log embedding + FAISS latency separately
"""

from app.embeddings.embedder import Embedder
from app.cache.faiss_index import FaissIndex
import time

embedder = Embedder()
index = FaissIndex(dimension=embedder.dimension)

# 100 diverse queries
queries = [
    "How do I reverse a list in Python?",
    "What is the capital of France?",
    "Explain quantum computing simply.",
    "Write a function to check palindromes.",
    "What is machine learning?",
    "How do I sort a dictionary by value in Python?",
    "What is the difference between a list and a tuple?",
    "How does garbage collection work in Python?",
    "Explain Python decorators with an example.",
    "How do I read a CSV file using Python?",
    "What is object-oriented programming?",
    "Explain inheritance in Java.",
    "What is the difference between an interface and an abstract class?",
    "How does Java garbage collection work?",
    "Write a Java program to find the largest element in an array.",
    "What is method overloading in Java?",
    "Explain exception handling in Java.",
    "What is the difference between ArrayList and LinkedList?",
    "How does a HashMap work internally?",
    "What is polymorphism in object-oriented programming?",
    "What is SQL normalization?",
    "Explain the difference between primary key and foreign key.",
    "What is a database index?",
    "Write a SQL query to find the second highest salary.",
    "What is the difference between WHERE and HAVING?",
    "Explain INNER JOIN and LEFT JOIN.",
    "What is database transaction isolation?",
    "What does ACID mean in databases?",
    "How does a B-tree index work?",
    "What is database normalization up to 3NF?",
    "What is an operating system?",
    "Explain the difference between a process and a thread.",
    "What is virtual memory?",
    "How does CPU scheduling work?",
    "Explain deadlock in operating systems.",
    "What are the four necessary conditions for deadlock?",
    "What is a context switch?",
    "Explain paging in operating systems.",
    "What is a semaphore?",
    "What is the difference between a mutex and a semaphore?",
    "What is TCP?",
    "Explain the TCP three-way handshake.",
    "What is the difference between TCP and UDP?",
    "How does DNS resolution work?",
    "What is an IP address?",
    "Explain IPv4 versus IPv6.",
    "What is HTTP?",
    "What is the difference between HTTP and HTTPS?",
    "How does a router forward packets?",
    "What is network congestion control?",
    "What is a binary search tree?",
    "Explain depth-first search.",
    "Explain breadth-first search.",
    "What is the time complexity of binary search?",
    "How does merge sort work?",
    "Explain quicksort.",
    "What is dynamic programming?",
    "Explain the difference between BFS and DFS.",
    "What is Dijkstra's shortest path algorithm?",
    "How does a hash table work?",
    "What is Docker?",
    "Explain the difference between a Docker image and container.",
    "How do I create a Dockerfile?",
    "What is Kubernetes used for?",
    "What is a REST API?",
    "Explain the difference between GET and POST requests.",
    "What is FastAPI?",
    "How do I create an endpoint in FastAPI?",
    "What is middleware in a web application?",
    "What is an API gateway?",
    "What is artificial intelligence?",
    "What is deep learning?",
    "Explain the difference between supervised and unsupervised learning.",
    "What is a neural network?",
    "What is a transformer model?",
    "How does attention work in transformers?",
    "What is an embedding?",
    "What are vector databases?",
    "What is semantic search?",
    "How does retrieval augmented generation work?",
    "What is FAISS?",
    "How does approximate nearest neighbor search work?",
    "What is cosine similarity?",
    "Explain Euclidean distance.",
    "What is a vector embedding?",
    "How do recommendation systems use embeddings?",
    "What is cache invalidation?",
    "What is a cache hit?",
    "What is a cache miss?",
    "Explain write-through caching.",
    "What is Git?",
    "How do I create a new Git branch?",
    "What is the difference between git merge and git rebase?",
    "How do I undo the last Git commit?",
    "How do I check which Git files are untracked?",
    "What is continuous integration?",
    "What is continuous deployment?",
    "Explain environment variables.",
    "What is a virtual environment in Python?",
    "How do I install dependencies from requirements.txt?",
    "What is the capital of Japan?",
    "Why is the sky blue?",
    "How does photosynthesis work?",
    "What is the boiling point of water?",
    "Explain how earthquakes occur.",
    "What causes tides?",
    "How does the human immune system work?",
    "What is the difference between renewable and nonrenewable energy?",
    "How does solar power generation work?",
    "Why do objects fall toward Earth?",
]

# Pad to 100 if needed
while len(queries) < 100:
    queries.append(f"Generic question number {len(queries) + 1}")

# Embed and index all 100
print("Embedding 100 queries...")
start = time.time()
for i, q in enumerate(queries):
    vec = embedder.embed(q)
    index.add(entry_id=i, vector=vec)
embed_time = (time.time() - start) * 1000
print(f"  Total embed time: {embed_time:.0f} ms ({embed_time/100:.1f} ms/query)")
print(f"  Index size: {index.size} vectors")

# 10 near-duplicate test queries (paraphrases of queries above)
test_pairs = [
    ("How can I reverse a Python list?", "How do I reverse a list in Python?"),
    ("What's France's capital city?", "What is the capital of France?"),
    ("Explain quantum computing in easy terms.", "Explain quantum computing simply."),
    ("Write a palindrome checker function.", "Write a function to check palindromes."),
    ("What is ML?", "What is machine learning?"),
    ("How to flip a list in Python?", "How do I reverse a list in Python?"),
    ("Capital of France?", "What is the capital of France?"),
    ("Quantum computing for beginners", "Explain quantum computing simply."),
    ("Check if string is palindrome", "Write a function to check palindromes."),
    ("Define machine learning", "What is machine learning?"),
]

print(f"\nSearching for 10 near-duplicates...")
for query, expected_match in test_pairs:
    vec, embed_ms = embedder.embed_timed(query)
    results, faiss_ms = index.search_timed(vec, top_k=3)

    top_id, top_score = results[0]
    matched_query = queries[top_id]

    status = "OK" if matched_query == expected_match else "CHECK"
    print(f"  [{status}] \"{query[:40]}...\"")
    print(f"         -> \"{matched_query[:40]}...\" (sim={top_score:.4f})")
    print(f"         embed={embed_ms:.1f}ms  faiss={faiss_ms:.3f}ms")

print(f"\nDone. Review the results above — every [OK] means the top result matched the expected query.")
print(f"Any [CHECK] means the top result was different — inspect manually to see if it's still sensible.")
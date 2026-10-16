Step 47 - RAG evaluation hardening

1) Back up existing ask.py and evaluation/evaluate.py.
2) Copy ask.py to the project root.
3) Copy evaluation/evaluate.py to the evaluation folder.
4) KEEP your existing evaluation/questions.json, evaluation/query_expansion_cache.json,
   data folder, and .env; these are NOT included in this package.
5) Run from project root:
   .\.venv\Scripts\python.exe evaluation/evaluate.py --only q16 q17 q18 q19

The evaluation defaults to at least 13 seconds between Gemini requests and
retries 503/temporary network failures at most twice. It never retries 429.
Results: evaluation/results.json and timestamped results_*.json.
Manual factual/citation review: evaluation/review_*.md.

If quota is already exhausted (429), wait for its reset. No program can bypass
an API provider's rate/daily quota. A 503 may remain after bounded retries.

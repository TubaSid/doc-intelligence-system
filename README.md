Document Intelligence System
============================

Overview
Built to ingest long form PDFs (Tesla 10-K) into a Pinecone vector index using local all-MiniLM-L6-v2 embeddings, then answer questions through a Groq LLM via FastAPI and LangGraph.

What's inside
* PDF ingestion with chunking (2000 chars, 400 overlap)
* Pinecone index (doc-intelligence) with 275 chunks
* Local embeddings (sentence-transformers) to avoid API limits
* Groq LLM: llama-3.1-8b-instant for routing, answering, and verification
* LangGraph multi-agent workflow (router, retriever, answerer, verifier, fallback)
* FastAPI server runs locally at http://localhost:8000 with /api/v1/query (reachable from the same machine unless you expose the port)

Quick start
1. Activate venv: `venv\Scripts\Activate.ps1`
2. Copy .env.example to .env and set GROQ_API_KEY, PINECONE_API_KEY
3. Ensure requirements are installed (includes pinecone, langchain-groq, sentence-transformers, python-multipart)
4. Build index (once): `python src/pipeline.py`
5. Run API: `python src/api/main.py`
6. Test: `python test_optimized.py`

Data
Place tesla_10k.pdf at data/raw/tesla_10k.pdf (not committed to repo), then run `python src/pipeline.py` to index it.

Test results (latest)
Command: `python test_optimized.py`
Outcome: 6/6 queries answered successfully
Average confidence: 50.0%
Sample queries and answers
- Q: How many vehicles delivered in 2023? | A: 1,808,581 delivered (Source 5)
- Q: What was Tesla total revenue in 2023? | A: $96,773 million (Source 1)
- Q: How much energy storage was deployed in 2023? | A: 14.72 GWh (Source 4)
- Q: What are the main risk factors mentioned? | A: Returns the risk factors section excerpts from the filing

API usage
POST http://localhost:8000/api/v1/query (from the host machine; use http://<host-ip>:8000/api/v1/query if accessing over the network)
Body: {"query": "How many vehicles delivered in 2023?"}

Notes
* Uses pinecone==8.0.0 (no pinecone-client)
* Embeddings are local so no token cost there
* Switch models in src/agents/nodes.py if you want a different Groq model
* Current LLM: llama-3.1-8b-instant (Groq)
* Dependencies listed in requirements.txt (includes python-multipart)

Dependencies (core)
python-multipart
pinecone==8.0.0
langchain-groq
sentence-transformers

Access over network
API listens on 0.0.0.0:8000; from another machine use http://<host-ip>:8000/api/v1/query

Contact
[linkedin.com/in/tubasid](https://linkedin.com/in/tubasid)
[tubaasid@gmail.com](mailto:tubaasid@gmail.com)

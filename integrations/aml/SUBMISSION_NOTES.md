# AML Submission Notes (paste into the form)

Fill these values in the "Apply for an Evaluation Key" form at
`agentmemoryleaderboard.ai` (button: Submit Evaluation Request):

| Form field | Value |
|---|---|
| Leaderboard track | **Academic leaderboard** |
| Academic evaluation method | **Submit GitHub code for platform deployment** |
| Work email | *(your email)* |
| Name | *(your name)* |
| Organization / Team | *(optional)* |
| System name | `CoreMem` |
| Version name | `0.12.0` |
| Public GitHub repository URL | `https://github.com/open-assistants-lab/CoreMem` |
| Submission notes and run instructions | paste the block below |

---

## Submission notes and run instructions (paste this)

```
CoreMem — zero-LLM memory retrieval for AI agents.

Docker entrypoint (repository root):
  docker build -t coremem-aml -f integrations/aml/Dockerfile .
  docker run -p 8000:8000 coremem-aml
The image builds CoreMem from this repository, pre-downloads the embedding
and cross-encoder models at build time (instant startup), and exposes the
AML Add/Search API on port 8000. Models are fully local — no API keys,
no network calls at query time.

API wrapper (implemented in integrations/aml/server.py, documented in
integrations/aml/README.md):
  POST /v1/memories/add     — user_id isolation, session_id grouping,
                              timestamp (Unix ms), success echo envelope
  POST /v1/memories/search  — {query, user_id, top_k, options?} →
                              data[{id, content, score, created_at}]
  GET  /health              — liveness

Method (original, open source, MIT):
Deterministic zero-LLM retrieval: hybrid search (FTS5 + embeddings) →
temporal query decomposition → RRF fusion (per-variant top-40 union for
preference queries) → cross-encoder rerank (MiniLM-L-6) → MMR session
diversity. Validated on LongMemEval-S (500 questions): 0.950 session
recall@5 baseline; temporal decomposition +0.037 and preference union
+0.033 session recall (all deltas measured on the S fixture).
```

---

## Reference checklist (verified against the live api-guide)

- [ ] Public GitHub repository: `https://github.com/open-assistants-lab/CoreMem` (public, in sync)
- [ ] Docker entrypoint documented: `integrations/aml/Dockerfile` (builds from the repo, models baked in, 15s startup)
- [ ] API wrapper instructions: `integrations/aml/README.md`
- [ ] Contract verified: `user_id` isolation, `session_id` grouping, `timestamp` (Unix ms), `success` echo envelope, `data[{id, content, score, created_at}]` responses
- [ ] Original-method disclosure: deterministic zero-LLM pipeline with validated retrieval improvements (temporal decomposition +0.037, preference union +0.033 on LongMemEval-S)
- [ ] No leaderboard key issued for the academic route — maintainers clone, build, and evaluate

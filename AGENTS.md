# CoreMem — Agent Context

## Eval Results — LongMemEval Oracle (500 questions, k=5, ollama-cloud:deepseek-v4-flash)

**Dataset**: `data/longmemeval_oracle.json` — 500 questions, ~2 sessions each  
**Results**: `eval_output/lme-oracle/results.json`

| Metric | `memorycore` | `memorycore_deep` | `memorycore_journal` |
|---|---|---|---|
| session_recall@5 | 0.938 | **0.951** | 0.666 |
| message_recall@5 | 0.754 | **0.854** | 0.600 |
| session_hit@5 | **0.972** | **0.972** | 0.748 |
| message_hit@5 | 0.904 | **0.951** | 0.726 |
| session_mrr | **0.972** | **0.972** | 0.748 |
| message_mrr | 0.702 | **0.798** | 0.564 |
| session_map | 0.938 | **0.951** | 0.666 |
| session_precision@5 | 0.475 | **0.479** | 0.306 |
| empty_retrieval_rate | **0.060** | **0.060** | 0.280 |
| abstention_false_positive_rate | **0.0** | **0.0** | **0.0** |
| context_chars_mean | 4937 | **3928** | 2776 |

**Key findings**:
- `memorycore_deep` is the best overall — highest on every message-level metric, ties `memorycore` on session metrics, uses 20% less context
- `memorycore` (zero-LLM) achieves 93.8% session recall — strong baseline with no API calls
- `memorycore_journal` empty_retrieval_rate (0.280) inflated by a 429-damaged batch (questions 207-324); healthy batches show ~16-21% empty rate
- `memorycore_journal` makes 1 LLM call per question (journal compilation); `memorycore_deep` makes 1 LLM call per question (query expansion); `memorycore` makes zero
- Zero pages (journal compilation failed): 105/500 (21%) — 87 from 429 rate-limit damage in batch 2, ~14 from plan validation in healthy batches (2.8%)
- All 3 modes abstain correctly on unanswerable questions (0% false positive rate)

### 20-Question Subset (for quick iteration)

| Metric | `memorycore` | `memorycore_deep` | `memorycore_journal` |
|---|---|---|---|
| session_recall@5 | 0.825 | **0.895** | 0.789 |
| message_recall@5 | 0.623 | 0.702 | **0.719** |
| session_hit@5 | **1.0** | **1.0** | 0.895 |
| message_hit@5 | **0.895** | 0.842 | 0.842 |
| session_mrr | 0.93 | **0.947** | 0.406 |
| session_map | 0.731 | **0.842** | 0.331 |
| empty_retrieval_rate | 0.05 | 0.05 | 0.05 |
| abstention_false_positive_rate | 0.0 | 0.0 | 0.0 |

Results saved in `eval_output/coremem-lme-full-all/results.json`

## Eval Results — LongMemEval S (500 questions, k=5, memorycore only)

**Dataset**: `data/longmemeval_s_cleaned.json` — 500 questions, ~48 sessions each, 265 MB JSON  
**Results**: `eval_output/lme-s/results.json`, `eval_output/lme-s/results.jsonl`  
**Note**: `memorycore` only — VM too small for cross-encoder (memorycore_deep/journal). Run on VM with 4 GB extra swap for `json.load`.

| Metric | S (memorycore) | Oracle (memorycore) |
|---|---|---|
| session_recall@5 | 0.865 | 0.938 |
| message_recall@5 | 0.670 | 0.754 |
| session_hit@5 | 0.968 | 0.972 |
| session_mrr | 0.968 | 0.972 |
| session_map | 0.831 | 0.938 |
| empty_retrieval_rate | 0.060 | 0.060 |
| abstention_false_positive_rate | 0.0 | 0.0 |
| mean_time_seconds | 165 | 80 |

**By question type:**

| Type | s@5 | m@5 | n |
|---|---|---|---|
| single-session-assistant | **1.000** | **0.857** | 56 |
| single-session-user | **0.969** | **0.898** | 70 |
| knowledge-update | 0.931 | 0.738 | 78 |
| single-session-preference | 0.867 | 0.544 | 30 |
| temporal-reasoning | 0.796 | 0.588 | 133 |
| multi-session | 0.779 | 0.539 | 133 |

**Key findings**:
- Zero-LLM `memorycore` holds at 86.5% session recall with ~48 sessions/question (vs 93.8% with ~2)
- `single-session-assistant` and `single-session-user` are near-perfect (0.97-1.0)
- `multi-session` and `temporal-reasoning` are hardest (0.78-0.80) — finding right sessions among ~48
- `memorycore_deep` (query expansion) and `memorycore_journal` not yet run on S — need bigger VM

## Full LongMemEval Dataset

Three 500-question variants downloaded from Hugging Face:

| File | Size | Sessions/question |
|---|---|---|
| `data/longmemeval_oracle.json` | 15 MB | ~2 |
| `data/longmemeval_s_cleaned.json` | 265 MB | ~48 |
| `data/longmemeval_m_cleaned.json` | 2.5 GB | ~475 |

All share the same 6 question types (same distribution as the 20-question subset).

## Remote VM Eval Plan

**Target**: `root@172.105.180.214` — 1 CPU, 939 MB RAM, 25 GB disk  
**SSH key**: `/Users/eddy/Library/Mobile Documents/com~apple~CloudDocs/SSH/rsa_linode_mc`  
**Project dir**: `~/coremem/` on VM  
**Eval output root**: `~/coremem/eval_output/lme-oracle/` on VM (19 GB free, plenty for oracle)  
**Too small for S/M** — oracle only (500 questions, ~2 sessions each).  
M variant estimated at ~500 MB per question (475 sessions × ~1 KB messages + vectors), so 500 questions would need ~250+ GB — not feasible on this VM.

Deploy and run in chunks via `--resume`:

```bash
rsync -az --delete --exclude '.venv' --exclude '.git' --exclude 'eval_output' \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.firecrawl' --exclude '.llm_cache' \
  /Users/eddy/Developer/Python/CoreMem/ root@172.105.180.214:~/coremem/

rsync -az /Users/eddy/Developer/Python/CoreMem/.env root@172.105.180.214:~/coremem/.env

ssh root@172.105.180.214 -t 'cd ~/coremem && source .venv/bin/activate && source .env && \
  uv run python3 scripts/eval_agent_journal_longmemeval.py data/longmemeval_oracle.json \
    --mode memorycore_journal --k 5 --limit 100 \
    --journal-llm-model ollama-cloud:deepseek-v4-flash \
    --progress --root ~/coremem/eval_output/lme-oracle --output ~/coremem/eval_output/lme-oracle/results.json --overwrite'
```

Chunking is automatic — re-run with `--resume` to continue after the VM times out (expected every ~2hrs on 1 CPU).  
Use `tmux` to keep session alive across disconnect:

```bash
tmux new-session -d -s coremem-eval 'bash -l'
tmux send-keys -t coremem-eval "ulimit -n 65536 && cd ~/coremem && source .venv/bin/activate && source .env" Enter
tmux send-keys -t coremem-eval "uv run python3 scripts/eval_agent_journal_longmemeval.py data/longmemeval_oracle.json \
    --mode memorycore_journal --k 5 --limit 100 \
    --journal-llm-model ollama-cloud:deepseek-v4-flash \
    --progress --root ~/coremem/eval_output/lme-oracle --output ~/coremem/eval_output/lme-oracle/results.json --overwrite" Enter
```

**Note**: `ulimit -n 65536` is needed to avoid `sqlite3.OperationalError: unable to open database file` — each per-question instance opens a SQLite DB, and the default 1024 file descriptor limit is hit around question 68.

Attach: `ssh root@172.105.180.214 -t 'tmux attach -t coremem-eval'`  
Check progress: `ssh root@172.105.180.214 'tail -20 ~/coremem/eval_output/lme-oracle/eval.log'`  
Sync results back: `rsync -az root@172.105.180.214:~/coremem/eval_output/lme-oracle/results.json eval_output/`

### Monitoring

Each question in the results now includes:
- `question_time_seconds` — wall clock per question
- `instance_disk_mb` — disk usage per question (hybriddb + vectors + journal)

The final results also include cumulative stats:
- `cumulative_time_seconds` / `mean_time_seconds`
- `cumulative_disk_mb` / `mean_disk_mb`

Check progress mid-run:
```bash
# Time and disk so far
ssh root@172.105.180.214 "python3 -c '
import json
c = json.load(open(\"~/coremem/eval_output/lme-oracle/results.json.checkpoint.json\"))
times = [r.get(\"question_time_seconds\",0) for m in c.get(\"modes\",{}).values() for r in m.get(\"results\",[]) if r.get(\"question_time_seconds\")]
disks = [r.get(\"instance_disk_mb\",0) for m in c.get(\"modes\",{}).values() for r in m.get(\"results\",[]) if r.get(\"instance_disk_mb\")]
print(f\"Completed: {len(c.get(\"completed_question_ids\",[]))}\")
print(f\"Total time: {sum(times)/60:.1f} min, Mean: {sum(times)/len(times):.1f}s\" if times else \"No times yet\")
print(f\"Total disk: {sum(disks):.0f} MB, Mean: {sum(disks)/len(disks):.1f} MB\" if disks else \"No disk yet\")
'"

# Disk usage of entire eval output
ssh root@172.105.180.214 "du -sh ~/coremem/eval_output/lme-oracle"
```

### S Run (scheduled)

The S variant (500 questions, ~48 sessions each) is scheduled to start 1 hour after the oracle run completes. It uses `--cleanup-instances` to delete per-question data after scoring, keeping only the results JSON. Run in chunks of 50:

```bash
# Attach to S run
ssh root@172.105.180.214 -t 'tmux attach -t coremem-eval-s'

# Resume S run after timeout
ssh root@172.105.180.214 -t 'cd ~/coremem && source .venv/bin/activate && source .env && \
  ulimit -n 65536 && uv run python3 scripts/eval_agent_journal_longmemeval.py data/longmemeval_s_cleaned.json \
    --mode memorycore_journal --k 5 --limit 50 \
    --journal-llm-model ollama-cloud:deepseek-v4-flash \
    --progress --root ~/coremem/eval_output/lme-s \
    --output ~/coremem/eval_output/lme-s/results.json --resume --cleanup-instances'
```

## Key Design Decisions

- **No verbatim compiler** — removed; only LLM compiler for daily journals
- **Daily pages use hybriddb timestamps** — `daily/{actual_date}.md`, not `datetime.now(UTC)`
- **`DEFAULT_AGENT_JOURNAL_MODEL`** = `"openai:gpt-4o-mini"` (ollama-cloud not in library default)
- **`--journal-llm-model` required** for `memorycore_journal` mode
- **Per-question haystack** — canonical LongMemEval setup
- **Shared `CrossEncoderReranker`** across per-question cores (avoids reloading model)
- **Resume/checkpoint** via sidecar `{output}.checkpoint.json`

## Eval CLI

```bash
# All modes (20-question subset)
uv run scripts/eval_agent_journal_longmemeval.py data/longmemeval_20_baseline_subset.json \
  --mode all --k 5 --journal-llm-model ollama-cloud:deepseek-v4-flash \
  --progress --root /tmp/coremem-lme --output results.json --overwrite

# Resume
uv run scripts/eval_agent_journal_longmemeval.py data/longmemeval_20_baseline_subset.json \
  --mode all --k 5 --journal-llm-model ollama-cloud:deepseek-v4-flash \
  --progress --root /tmp/coremem-lme --output results.json --resume

# Full oracle (500 questions, ~2 sessions each) — all 3 modes
uv run scripts/eval_agent_journal_longmemeval.py data/longmemeval_oracle.json \
  --mode all --k 5 --journal-llm-model ollama-cloud:deepseek-v4-flash \
  --progress --root /tmp/coremem-lme-oracle --output results.json --overwrite

# Chunked (first 100, resume for more)
uv run scripts/eval_agent_journal_longmemeval.py data/longmemeval_oracle.json \
  --mode all --k 5 --limit 100 --journal-llm-model ollama-cloud:deepseek-v4-flash \
  --progress --root /tmp/coremem-lme-oracle --output results.json --overwrite
```

## Tests

```bash
uv run python3 -m pytest tests/ -q   # 98 pass
```
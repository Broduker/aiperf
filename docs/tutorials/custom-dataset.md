---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Custom Dataset Guide
---

# Custom Dataset Guide

Benchmark LLMs with your own data using single-turn requests, multi-turn conversations, random sampling, or production trace replay.

## Overview

AIPerf supports these custom dataset types for benchmarking with your own data:

| Dataset Type | Best For | Multi-Turn | Timing Control | Random Sampling |
|-------------|----------|-----------|---------------|-----------------|
| **Single Turn** | Independent single requests | No | Yes | No |
| **Multi Turn** | Conversations with context | Yes | Yes (per turn) | No |
| **Random Pool** | Load testing with variety | No | No | Yes |
| **Mooncake / Bailian / Baseten Trace** | Production trace replay | Yes | Yes | No |

**Single Turn, Multi Turn, and Random Pool support:**
- Client-side batching
- Automatic media handling: local files are converted to base64 format, while remote URLs are sent directly to the API

Trace replay requests are text-only, so client-side batching and media handling do not apply. See [Trace Replay](../benchmark-modes/trace-replay.md) and [Baseten Trace Replay](baseten-trace.md).

---

## Server Setup

Start a vLLM server for testing:

```bash
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 8000 &
```

Verify the server is ready:
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "test"}],
    "max_tokens": 10
  }' | jq
```

---

## Single-Turn Datasets

### Resume completed synthesis requests

For a long-running synthesis job, set `resume_from` on a path-backed
`single_turn` dataset to reuse successful requests from earlier RAW exports:

```yaml
benchmark:
  models: [my-model]
  endpoint:
    type: chat
    urls: [http://localhost:8000/v1/chat/completions]
  datasets:
    - name: main
      type: file
      format: single_turn
      path: /data/input.jsonl
      sampling: sequential
      resume_from:
        - /data/previous-run
        - /data/other-run/raw_records_worker_0.jsonl.zst
  phases:
    - name: profiling
      type: concurrency
      concurrency: 32
      requests: 1000000
  artifacts:
    dir: /data/resumed-run
    raw: true
```

Every input row must have a unique, explicit `session_id` and represent one
independent turn. Recovery scans plain or zstd-compressed RAW JSONL files,
including `raw_records_*.jsonl[.zst]` files below supplied directories. It streams
input and stores the success index in SQLite on disk. It bypasses the mmap
dataset cache so changes in history cannot reuse an outdated filtered dataset.

A hit requires the same `session_id` **and complete request body** (model,
messages, stream setting, generation parameters and extra fields). A historical
profiling record must have HTTP 200, no error, `finish_reason: stop`, and nonempty
text for every requested choice. Failed, empty, truncated, unsupported or missing
results are retried. One successful duplicate wins over later failures. Invalid
complete JSONL records fail preparation; an unterminated last line is excluded.
For active plain JSONL writers, only complete lines within the initial file size
are considered. Compressed files must be closed before use.

Recovery supports one chat model, one sequential file dataset and one profiling
concurrency or rate phase, including adaptive scale. Warmup, accuracy evaluation
and cache busting cannot be combined with recovery. The phase sends at most one
pass of pending samples; a smaller request/session cap or duration can stop it
earlier. An all-cached input completes with zero requests. Adaptive discovery
starts afresh; historical performance statistics are not merged into this run.

The artifact directory contains `result_cache/report.json` with scanned,
unique, duplicate, ignored, reused and pending counts, plus
`result_cache/reused.jsonl` mapping reused IDs to historical file and line.
Historical outputs stay in place and must be retained alongside new RAW output
for downstream synthesis merging. Paths must be readable by the dataset manager;
on Kubernetes mount the history PVC there and create a new `AIPerfJob` with a new
artifact directory. Freeze or stop an old writer before a final handover if
duplicate work after the snapshot is unacceptable.

### Continue writing a persistent result file

Set `cache_dir` to keep complete successful responses in one durable JSONL file,
similar to EvalScope prediction caching. It works on the first run without
`resume_from`; subsequent runs reuse the same cache directory automatically:

```yaml
benchmark:
  models: [my-model]
  endpoint:
    type: chat
    urls: [http://localhost:8000/v1/chat/completions]
  datasets:
    - name: main
      type: file
      format: single_turn
      path: /data/input.jsonl
      sampling: sequential
      cache_dir: /data/synthesis/cache
      resume_from: [/data/previous-run]
  phases:
    - name: profiling
      type: concurrency
      concurrency: 32
      requests: 1000000
  artifacts:
    dir: /data/synthesis/run
    raw: true
```

`cache_dir/results.jsonl` contains complete successful **RAW records**, including
request payloads and responses, from history and the current run. New successes
are appended during periodic RAW flushes and at finalization; they do not wait
for a manual merge. HTTP failures, empty answers and truncated generations are
not cached. One record per `(session_id, request-body fingerprint)` is retained;
changing a prompt or generation parameters creates a distinct result. Output is
in completion/import order, not input order. This is an AIPerf RAW export, not
the original dataset's business-specific output schema.

When `resume_from` is supplied, successful history is streamed into this file.
After import, history files are no longer needed for cached answers. Remove
`resume_from` on later runs to avoid scanning them again. To cover an entire
original dataset, supply its full converted input and import all earlier result
directories, including successes excluded from an old `remaining.jsonl` snapshot.

The artifact directory may also be reused in this mode. RAW processors use
unique staging filenames so initialization cannot truncate historical inputs.
Per-run reports and performance exports can be replaced; `cache_dir` holds the
durable synthesis answers. Recovery reports remain preparation-time accounting,
and historical performance statistics are not included in new-run metrics.

The disk-backed SQLite index stores IDs, fingerprints and a byte checkpoint,
never response bodies in memory. Restarts read this index rather than scanning
all historical response text again. Appends are serialized with a file lock across
record processors. After interruption, complete unindexed lines are recovered
and an incomplete final line is removed before appending. Keep `results.jsonl`
and its SQLite index together; a missing index can be rebuilt from the JSONL.
Use a dedicated cache directory for each logical dataset. The filesystem must
support cross-process file locking and SQLite transactions. Do not run two
independent jobs against the same cache: they can both schedule the same pending
sample even though successful output is deduplicated.

On Kubernetes, `cache_dir` must be an absolute PVC path mounted at the same
location in the dataset-manager and every record-processor pod, rather than the
pod-local `/results` directory. Stop the old job and flush its records before
handover, retain the PVC, then recreate the job with the same cache directory
and the desired concurrency configuration. Deleting a Kubernetes job alone does
not implement recovery. Finishing one pass also does not guarantee all samples
succeeded: restart against the same cache to retry remaining failures.

Each line represents one independent single-turn request.

### When to Use

Use single_turn when you need **deterministic, sequential execution** where requests always run in the exact order they appear in the file:

- **Debugging**: Test specific prompts in a known sequence
- **Regression testing**: Same input file → same output order every time
- **Timing control**: Schedule requests with precise timestamps or delays
- **Predictable testing**: Know exactly which request runs when

**Execution:** Sequential by default (request 1, then 2, then 3, etc.)
**Input:** Single JSONL file only

### Large-file streaming load

For a path-backed `single_turn` JSONL file that is too large to materialize as a
Python dataset, enable the dataset-level streaming loader:

```yaml
benchmark:
  dataset:
    type: file
    format: single_turn
    path: /data/prompts.jsonl
    sampling: sequential
    streaming: true
    streaming_batch_size: 256
```

`streaming: true` parses records incrementally and writes bounded batches into
the mmap backing store. The full prompt and media payloads are not retained in
the DatasetManager's Python conversation list. The timing service still receives
conversation metadata for scheduling, so this mode is intended for independent
single-turn records and does not eliminate metadata or mmap index storage.

The mode requires a path-backed `single_turn` dataset with sequential sampling.
Records that share a `session_id` must be contiguous; a session ID that reappears
after another session is rejected. It is not a replacement for trace replay,
random sampling, or multi-turn datasets. `streaming_batch_size` controls the
bounded write batch and defaults to 256.

Setting `resume_from` automatically selects this same streaming loader, even
when `streaming` is omitted or false. `CacheManager` filters historical successes
within each bounded batch before it is written to mmap. `streaming_batch_size`
also controls recovery batches, including batches in which every sample is
cached. Recovery requires unique explicit IDs and one turn per ID; ordinary
streaming continues to group contiguous records sharing a session ID.

### Basic Text Example

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
cat > prompts.jsonl << 'EOF'
{"text": "What is machine learning?"}
{"text": "Explain neural networks."}
{"text": "How does backpropagation work?"}
{"text": "What are transformers?"}
{"text": "Define reinforcement learning."}
EOF

aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --input-file prompts.jsonl \
    --custom-dataset-type single_turn \
    --streaming \
    --url localhost:8000 \
    --concurrency 2 \
    --request-count 10
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Output:**
```


                                     NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃               Metric ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p50 ┃      std ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│  Time to First Token │    19.99 │    12.53 │    49.62 │    48.89 │    42.24 │    13.93 │    12.92 │
│                 (ms) │          │          │          │          │          │          │          │
│ Time to Second Token │     3.81 │     2.01 │     8.25 │     7.94 │     5.15 │     3.36 │     1.62 │
│                 (ms) │          │          │          │          │          │          │          │
│ Time to First Output │    19.99 │    12.53 │    49.62 │    48.89 │    42.24 │    13.93 │    12.92 │
│           Token (ms) │          │          │          │          │          │          │          │
│ Request Latency (ms) │ 2,940.39 │ 1,536.67 │ 7,319.35 │ 7,034.86 │ 4,474.42 │ 2,239.67 │ 1,611.04 │
│  Inter Token Latency │     3.52 │     3.47 │     3.64 │     3.63 │     3.56 │     3.50 │     0.05 │
│                 (ms) │          │          │          │          │          │          │          │
│         Output Token │   284.54 │   274.60 │   288.35 │   288.33 │   288.13 │   285.38 │     3.98 │
│  Throughput Per User │          │          │          │          │          │          │          │
│    (tokens/sec/user) │          │          │          │          │          │          │          │
│      Output Sequence │   833.40 │   438.00 │ 2,106.00 │ 2,022.21 │ 1,268.10 │   626.50 │   465.81 │
│      Length (tokens) │          │          │          │          │          │          │          │
│       Input Sequence │     5.00 │     4.00 │     7.00 │     7.00 │     7.00 │     5.00 │     1.10 │
│      Length (tokens) │          │          │          │          │          │          │          │
│         Output Token │   527.06 │      N/A │      N/A │      N/A │      N/A │      N/A │      N/A │
│           Throughput │          │          │          │          │          │          │          │
│         (tokens/sec) │          │          │          │          │          │          │          │
│   Request Throughput │     0.63 │      N/A │      N/A │      N/A │      N/A │      N/A │      N/A │
│       (requests/sec) │          │          │          │          │          │          │          │
│        Request Count │    10.00 │      N/A │      N/A │      N/A │      N/A │      N/A │      N/A │
│           (requests) │          │          │          │          │          │          │          │
└──────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

CLI Command: aiperf profile --model 'Qwen/Qwen3-0.6B' --endpoint-type 'chat' --input-file
'prompts.jsonl' --custom-dataset-type 'single_turn' --streaming --url 'localhost:8000' --concurrency
2
Benchmark Duration: 15.81 sec
CSV Export:
artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency2/profile_export_aiperf.csv
JSON Export:
artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency2/profile_export_aiperf.json
Log File: artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency2/logs/aiperf.log
```

### Inline alternative

Same content as `prompts.jsonl`, embedded in the AIPerf YAML config:

```yaml
benchmark:
  model: Qwen/Qwen3-0.6B
  endpoint:
    url: http://localhost:8000
    type: chat
  dataset:
    type: file
    format: single_turn
    records:
      - {text: "What is machine learning?"}
      - {text: "Explain neural networks."}
      - {text: "How does backpropagation work?"}
      - {text: "What are transformers?"}
      - {text: "Define reinforcement learning."}
  phases:
    type: concurrency
    concurrency: 2
    requests: 100
```

See [Inline Datasets](inline-datasets.md) for the full feature reference.

### Per-Request Output Length

Control the maximum output tokens per request using the `output_length` field:

```bash
cat > prompts_with_osl.jsonl << 'EOF'
{"text": "Write a haiku about mountains.", "output_length": 50}
{"text": "Explain quantum computing in detail.", "output_length": 500}
{"text": "What is 2+2?", "output_length": 10}
{"text": "Summarize machine learning."}
EOF

aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --input-file prompts_with_osl.jsonl \
    --custom-dataset-type single_turn \
    --streaming \
    --url localhost:8000 \
    --osl 200 \
    --request-count 10
```

**Precedence:** Per-line `output_length` takes priority over the global `--osl` flag. Lines without `output_length` fall back to `--osl` if set (200 in this example), or let the server decide the output length.

The `output_length` field also works per-turn in multi_turn datasets.

### Per-Request `extra`

Send vendor-specific or sampling parameters per request via the `extra` field. The dict is shallow-merged into the top of the request body at dispatch. Per-line keys win over `--extra-inputs`:

```bash
cat > prompts_with_extra.jsonl << 'EOF'
{"text": "Brainstorm a haiku.", "extra": {"temperature": 1.2, "top_p": 0.9}}
{"text": "Explain quantum computing.", "extra": {"temperature": 0.2, "seed": 42}}
{"text": "Summarize ML.", "extra": {"min_tokens": 50, "ignore_eos": true}}
EOF
```

The `extra` field also works per-turn in multi_turn datasets.

---

## Multi-Turn Datasets

Each entry represents a complete conversation with multiple turns.

### When to Use

Use multi_turn when you need **conversations with context** where each turn builds on previous turns in the conversation:

- **Chat testing**: Test conversational AI that maintains context across turns
- **Realistic interactions**: Simulate real user conversations with follow-up questions
- **Task completion**: Test multi-step tasks that require conversation history

**Execution:** Sequential within each conversation (turn 1, then 2, then 3, etc.), but multiple conversations run concurrently
**Input:** Single JSONL file only

### Basic Conversation

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
cat > conversations.jsonl << 'EOF'
{"session_id": "chat_1", "turns": [{"text": "What is machine learning?"}, {"text": "Can you give me an example?"}]}
{"session_id": "chat_2", "turns": [{"text": "Explain neural networks."}, {"text": "How do they differ from traditional algorithms?"}, {"text": "Which architecture for image classification?"}]}
EOF

aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --input-file conversations.jsonl \
    --custom-dataset-type multi_turn \
    --streaming \
    --url localhost:8000 \
    --concurrency 2 \
    --request-count 10
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Output:**
```
                                     NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃                 Metric ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p50 ┃    std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│    Time to First Token │    23.17 │    11.83 │    56.70 │    55.34 │    43.06 │    18.00 │  13.66 │
│                   (ms) │          │          │          │          │          │          │        │
│   Time to Second Token │     4.77 │     2.29 │    15.41 │    14.65 │     7.73 │     3.44 │   3.74 │
│                   (ms) │          │          │          │          │          │          │        │
│   Time to First Output │    23.17 │    11.83 │    56.70 │    55.34 │    43.06 │    18.00 │  13.66 │
│             Token (ms) │          │          │          │          │          │          │        │
│   Request Latency (ms) │ 2,008.84 │ 1,348.13 │ 3,045.04 │ 3,007.53 │ 2,669.92 │ 2,082.32 │ 572.34 │
│    Inter Token Latency │     3.50 │     3.13 │     3.67 │     3.67 │     3.62 │     3.52 │   0.14 │
│                   (ms) │          │          │          │          │          │          │        │
│           Output Token │   286.03 │   272.35 │   319.58 │   316.89 │   292.60 │   283.77 │  12.33 │
│    Throughput Per User │          │          │          │          │          │          │        │
│      (tokens/sec/user) │          │          │          │          │          │          │        │
│ Output Sequence Length │   565.60 │   380.00 │   838.00 │   826.57 │   723.70 │   581.50 │ 150.96 │
│               (tokens) │          │          │          │          │          │          │        │
│  Input Sequence Length │   379.80 │     5.00 │ 1,331.00 │ 1,287.80 │   899.00 │   203.00 │ 438.88 │
│               (tokens) │          │          │          │          │          │          │        │
│           Output Token │   533.83 │      N/A │      N/A │      N/A │      N/A │      N/A │    N/A │
│             Throughput │          │          │          │          │          │          │        │
│           (tokens/sec) │          │          │          │          │          │          │        │
│     Request Throughput │     0.94 │      N/A │      N/A │      N/A │      N/A │      N/A │    N/A │
│         (requests/sec) │          │          │          │          │          │          │        │
│          Request Count │    10.00 │      N/A │      N/A │      N/A │      N/A │      N/A │    N/A │
│             (requests) │          │          │          │          │          │          │        │
└────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴────────┘

CLI Command: aiperf profile --model 'Qwen/Qwen3-0.6B' --endpoint-type 'chat' --input-file
'conversations.jsonl' --custom-dataset-type 'multi_turn' --streaming --url 'localhost:8000'
--concurrency 2
Benchmark Duration: 10.60 sec
CSV Export:
artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency2/profile_export_aiperf.csv
JSON Export:
artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency2/profile_export_aiperf.json
Log File: artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency2/logs/aiperf.log
```

**Key Points:**
- Each turn includes full conversation history
- Turns execute sequentially within each conversation
- Multiple conversations run concurrently (up to `--concurrency`)
- Each turn supports `output_length` and `extra` (same semantics as single_turn — vendor extras shallow-merged into the top of the wire body, latest turn wins for chat-style endpoints)

### Inline alternative

```yaml
benchmark:
  model: Qwen/Qwen3-0.6B
  endpoint:
    url: http://localhost:8000
    type: chat
  dataset:
    type: file
    format: multi_turn
    records:
      - session_id: chat_1
        turns:
          - {text: "What is machine learning?"}
          - {text: "Can you give me an example?"}
      - session_id: chat_2
        turns:
          - {text: "Explain neural networks."}
          - {text: "How do they differ from traditional algorithms?"}
          - {text: "Which architecture for image classification?"}
  phases:
    type: concurrency
    concurrency: 2
    requests: 100
```

### Persistent System Prompt

To apply a system prompt to every turn of a conversation, author it as a leading turn with `"role": "system"`. AIPerf lifts that turn into the conversation-level system message, so it is prepended to every turn's request rather than dispatched as its own (user-less) request:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
cat > system_prompt.jsonl << 'EOF'
{"session_id": "chat_1", "turns": [{"role": "system", "text": "You are a terse assistant. Answer in one sentence."}, {"text": "What is machine learning?"}, {"text": "Give me an example."}]}
EOF

aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --input-file system_prompt.jsonl \
    --custom-dataset-type multi_turn \
    --streaming \
    --url localhost:8000 \
    --concurrency 2 \
    --request-count 10
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Behavior:**
- The system prompt persists across all turns and is not counted as a turn (the example above runs 2 turns, not 3).
- Only a **leading**, **text-only** system turn is hoisted. A `system` turn that appears mid-conversation, one that carries image/audio/video media, or one that sets dispatch-time fields (`timestamp`, `delay`, `output_length`, `extra`) stays a normal turn.
- Hoisting only takes effect on endpoints that send a system message (`chat`, `responses`, `messages`, `chat_embeddings`). On other endpoints (e.g. `completions`) the leading system turn is left in place, so it is dispatched as a normal turn rather than being silently dropped.
- The same form works in the inline `records` config.

---

## Random Pool Datasets

Randomly sample from one or more data pools for varied request patterns.

### When to Use

Use random_pool when you need **random sampling with replacement** for unpredictable, varied request patterns:

- **Load testing**: Generate diverse request patterns with variety
- **Production simulation**: Model real-world workloads where requests vary
- **Stress testing**: Test system behavior under mixed input patterns
- **Multiple data sources**: Combine files from a directory (each file becomes a pool)

**Execution:** Random sampling with replacement (same entry can be selected multiple times)
**Input:** Single JSONL file OR directory of multiple JSONL files
**Note:** Does NOT support timing control or multi-turn conversations

### Basic Single-File Sampling

<!-- aiperf-run-vllm-default-openai-endpoint-server weight=250 -->
```bash
cat > pool.jsonl << 'EOF'
{"text": "What is machine learning?"}
{"text": "Explain neural networks."}
{"text": "How does backpropagation work?"}
{"text": "What are transformers?"}
{"text": "Define reinforcement learning."}
{"text": "What is transfer learning?"}
{"text": "Explain gradient descent."}
{"text": "What are GANs?"}
EOF

aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --input-file pool.jsonl \
    --custom-dataset-type random_pool \
    --num-conversations 50 \
    --streaming \
    --concurrency 4 \
    --random-seed 42 \
    --url localhost:8000
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Output:**
```
                                     NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃              Metric ┃      avg ┃      min ┃       max ┃      p99 ┃      p90 ┃      p50 ┃      std ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ Time to First Token │    17.73 │    12.25 │     53.21 │    53.17 │    19.85 │    14.63 │     9.90 │
│                (ms) │          │          │           │          │          │          │          │
│      Time to Second │     3.73 │     2.20 │     10.38 │     7.68 │     4.08 │     3.66 │     1.10 │
│          Token (ms) │          │          │           │          │          │          │          │
│       Time to First │    17.73 │    12.25 │     53.21 │    53.17 │    19.85 │    14.63 │     9.90 │
│   Output Token (ms) │          │          │           │          │          │          │          │
│     Request Latency │ 3,321.54 │ 1,356.57 │ 10,393.82 │ 9,063.81 │ 5,372.92 │ 2,917.73 │ 1,644.46 │
│                (ms) │          │          │           │          │          │          │          │
│ Inter Token Latency │     3.81 │     3.53 │      4.17 │     4.15 │     3.97 │     3.79 │     0.12 │
│                (ms) │          │          │           │          │          │          │          │
│        Output Token │   262.66 │   239.55 │    283.24 │   279.36 │   270.36 │   264.13 │     8.25 │
│ Throughput Per User │          │          │           │          │          │          │          │
│   (tokens/sec/user) │          │          │           │          │          │          │          │
│     Output Sequence │   861.02 │   369.00 │  2,615.00 │ 2,255.83 │ 1,306.40 │   766.00 │   404.28 │
│     Length (tokens) │          │          │           │          │          │          │          │
│      Input Sequence │     5.00 │     4.00 │      7.00 │     7.00 │     6.10 │     5.00 │     0.96 │
│     Length (tokens) │          │          │           │          │          │          │          │
│        Output Token │ 1,007.36 │      N/A │       N/A │      N/A │      N/A │      N/A │      N/A │
│          Throughput │          │          │           │          │          │          │          │
│        (tokens/sec) │          │          │           │          │          │          │          │
│  Request Throughput │     1.17 │      N/A │       N/A │      N/A │      N/A │      N/A │      N/A │
│      (requests/sec) │          │          │           │          │          │          │          │
│       Request Count │    50.00 │      N/A │       N/A │      N/A │      N/A │      N/A │      N/A │
│          (requests) │          │          │           │          │          │          │          │
└─────────────────────┴──────────┴──────────┴───────────┴──────────┴──────────┴──────────┴──────────┘

CLI Command: aiperf profile --model 'Qwen/Qwen3-0.6B' --endpoint-type 'chat' --input-file
'pool.jsonl' --custom-dataset-type 'random_pool' --num-conversations 50 --streaming --concurrency 4
--random-seed 42 --url 'localhost:8000'
Benchmark Duration: 42.74 sec
CSV Export:
artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency4/profile_export_aiperf.csv
JSON Export:
artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency4/profile_export_aiperf.json
Log File: artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency4/logs/aiperf.log
```

**Behavior:**
- Randomly samples 50 requests from 8-entry pool
- Sampling with replacement (entries can repeat)
- Use `--random-seed` for reproducibility

### Multimodal batch sizes

For multimodal workloads, `random_pool` supports per-modality batch-size flags that control how many items are packed into each request:

| Flag | Field | Default | Notes |
|---|---|---|---|
| `--prompt-batch-size N` | `prompt_batch_size` | 1 | Text items per request. Set to `0` to disable text inputs entirely. |
| `--image-batch-size N` | `image_batch_size` | 1 | Images per request. Set to `0` to disable image inputs. |
| `--audio-batch-size N` | `audio_batch_size` | 1 | Audio items per request. Set to `0` to disable audio inputs. |
| `--video-batch-size N` | `video_batch_size` | 1 | Video items per request. Set to `0` to disable video inputs. |

Setting a batch size to `0` for a modality that is absent from the pool is a no-op: it neither suppresses anything nor counts as batching, so `--image-batch-size 0` against a single-file text-only pool leaves that pool sampled normally. A configuration where *every* modality is either absent from the pool or set to `0` produces empty requests and is rejected, since there is nothing left to send. Directory input rejects any batch size other than `1` regardless of which modalities are present, because pool contents are not known at config time -- see below.

These flags are only valid with `format: random_pool` (set via `--custom-dataset-type random_pool` on the CLI, or `format: random_pool` directly in a YAML dataset config -- either selects the same format). Using them with other file dataset formats (e.g. `mooncake_trace`) is an error.

Batch sizes other than 1 are also rejected outright whenever batching would discard pool identity: (1) multiple separately named pools -- directory input (multiple files, e.g. `queries.jsonl` / `passages.jsonl`) or inline YAML `records:` with multiple top-level keys (e.g. `records: {queries: [...], passages: [...]}`) both flatten into one anonymous pool per modality, discarding the names that name-sensitive endpoints (e.g. rankings) depend on; (2) a single pool whose entries embed named `Text`/`Image`/`Audio`/`Video` objects, or `Image` objects carrying `uuids` (vLLM cache-reuse IDs) -- reachable from a single file or a single-key inline `records:` list, not just directory input. Batch sizes only apply to a single unnamed pool with unnamed entries.

The directory case is caught up front, before the benchmark starts. The remaining shapes are only visible once the pool has been parsed, so they surface when the dataset is built.

Batching does not preserve per-entry associations across modalities. If a pool entry pairs a specific text with a specific image, a batch size > 1 flattens each modality into an independent pool and samples from them separately, so the original text-image pairing is not preserved in the resulting request. Use `single_turn` instead of `random_pool` if exact pairing must be preserved.

The `pool.jsonl` file above is text-only, so `--image-batch-size` would have no images to sample. A multimodal `random_pool` needs an `image` field in the pool entries:

```bash
cat > multimodal_pool.jsonl << 'EOF'
{"text": "Describe what you see in this image.", "image": "https://example.com/img1.png"}
{"text": "What objects are visible here?", "image": "https://example.com/img2.png"}
{"text": "Summarize the scene.", "image": "https://example.com/img3.png"}
EOF

aiperf profile \
    --model Qwen/Qwen3-VL-7B \
    --endpoint-type chat \
    --input-file multimodal_pool.jsonl \
    --custom-dataset-type random_pool \
    --image-batch-size 2 \
    --num-conversations 50 \
    --streaming \
    --concurrency 4 \
    --url localhost:8000
```

### Inline alternative (multi-pool)

```yaml
benchmark:
  model: Qwen/Qwen3-0.6B
  endpoint:
    url: http://localhost:8000
    type: chat
  dataset:
    type: file
    format: random_pool
    sampling: random
    records:
      queries:
        - {text: "What is your refund policy?", type: random_pool}
        - {text: "How do I reset my password?", type: random_pool}
      passages:
        - {text: "Refunds are processed within 5 business days.", type: random_pool}
        - {text: "Click 'Forgot password' on the login page.", type: random_pool}
  phases:
    type: concurrency
    concurrency: 2
    requests: 50
```

---

## Adding a System Prompt

`--system-prompt` and `--system-prompt-file` attach a fixed system message to every
conversation. Unlike `--shared-system-prompt-length`, which generates synthetic filler of a
target token length, these take the **exact text** — so prefix-cache hit rates and TTFT
reflect the system prompt your deployment actually sends.

They work with synthetic, public, and structured file-based datasets. The one exclusion is
datasets that author their own request payloads — see the note at the end of this section.

```bash
# Inline, for short prompts
aiperf profile -m Qwen/Qwen3-0.6B --url http://localhost:8000 \
    --input-file ./data.jsonl \
    --system-prompt "You are a terse assistant."

# From a file, for real production prompts
aiperf profile -m Qwen/Qwen3-0.6B --url http://localhost:8000 \
    --input-file ./data.jsonl \
    --system-prompt-file ./prod_system.txt
```

Or in YAML:

```yaml
dataset:
  type: file
  path: ./data.jsonl
  system_prompt_file: ./prod_system.txt
```

Behavior worth knowing:

- **Tokens are additive.** With `--isl 1000` and a 350-token system prompt, the request
  carries roughly 1350 tokens: `--isl` continues to size the generated user prompt only.
- **A dataset's own system message is kept.** If the dataset already authors one, your text
  is prepended to it, separated by a blank line, and both are sent as a single system
  message. Repeated system roles are mishandled by many OpenAI-compatible servers.
- **The file is read once at startup**, so a missing or unreadable path fails immediately
  rather than mid-benchmark. Paths containing a symlinked component are rejected — note that
  on macOS this includes anything under `/tmp` or `$TMPDIR`, since `/var` is itself a symlink.
- **Supported on `chat`, `responses`, `messages`, and `chat_embeddings`.** Endpoints with no
  system role (`completions`, `embeddings`, the rankings endpoints) reject the option at
  startup rather than silently dropping it.
- **Not supported on datasets that author their own request payloads** — `raw_payload`,
  `inputs_json`, and `mooncake_trace` in payload mode. Those payloads are sent to the server
  byte-for-byte, bypassing the formatting step that puts the system message on the wire, so
  the prompt would be silently dropped. AIPerf rejects the combination at startup. Either
  author the system message directly into the trace payloads, or use a dataset type that
  produces structured turns (`single_turn`, `multi_turn`, `dag_jsonl`).
- **Mutually exclusive** with `--shared-system-prompt-length` and with
  `--num-prefix-prompts`/`--prefix-prompt-length`, all of which fill the same slot. Setting
  two of them is rejected at startup rather than one taking precedence, so a misconfigured
  run fails before sending any requests. It does combine with
  `--user-context-prompt-length` for a two-tier shared/per-session structure.

> **Note:** `--num-prefix-prompts` and `--prefix-prompt-length` apply only to synthetic
> datasets, and AIPerf rejects them at startup when combined with `--input-file` or
> `--public-dataset` rather than ignoring them. `--system-prompt` has no such restriction.
>
> The same rule now covers every synthetic-generation flag: the media *shape* options
> (`--image-width-mean`/`--image-height-mean`/`--image-format`, `--audio-length-mean`,
> `--audio-format`, `--audio-sample-rates`, the `--video-*` family) and the
> `--rankings-*` options. A file or public dataset supplies its own media and passages,
> so these previously did nothing at all; they are now an error. The per-modality
> batch-size flags remain supported with `--custom-dataset-type random_pool`, where they
> control how many items are packed per request -- but only when `--input-file` points
> at a single random-pool file, not a directory: a directory forms a separately named
> pool per file, and batching across pools has no defined meaning.

---

## Related

- [Multi-Turn Conversations](multi-turn.md) - Multi-turn conversation benchmarking
- [Conversation Context Mode](../reference/conversation-context-mode.md) - How conversation history accumulates in multi-turn

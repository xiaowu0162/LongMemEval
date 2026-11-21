# Evaluate a Loom memory service with LongMemEval

This optional adapter feeds LongMemEval histories to a
[Loom](https://github.com/ClickHouse/loom) memory service and writes the native
`question_id`/`hypothesis` JSONL interface. Grade that file with the unchanged
`src/evaluation/evaluate_qa.py`. No shared evaluator or baseline is modified.

## Treatment and outputs

The reader is adapted from the native facts/CoT template. It adds temporal
precedence, counting distinct instances, coreference, date arithmetic and
optional structured aggregate guidance. Those instructions can affect answers
independently of indexing and retrieval; this is not native-reader parity.
For non-reasoning models the reader requests `temperature=0` and
`max_tokens=800`. Its reasoning-model branch requests no output-token cap.

The script reports custom any-session coverage, all-evidence-session coverage
and answer-string presence diagnostics. They include all question outcomes,
including abstention cases; they are not the native retrieval metric, which
excludes abstention cases. It does not emit native `retrieval_results` logs for
`run_generation.py` or `print_retrieval_metrics.py`.

No empirical results are included. Offline checks do not establish released
server compatibility, live authorization, extraction quality, model quality,
latency, token savings or a comparison with another memory system.

## Prerequisites

- Use an isolated Loom test deployment that supports
  `memory.set_from_messages` with inline `defer: false` and `memory.search`.
  The adapter uses org `dev`, agent `lme-loom` and a random `lme-...` namespace
  per question. Existing grants must authorize these scopes and operations;
  the adapter does not provision or broaden grants.
- Follow the upstream [evaluation-only environment](../README.md#evaluation-only)
  and [dataset instructions](../README.md#data). Use the cleaned dataset
  without substituting another version during a comparison.
- Install the adapter dependency with `python -m pip install -r loom/requirements.txt`.
- Supply your own `LOOM_TOKEN` and `OPENAI_API_KEY` through the environment.
  The key is used by the reader and the official evaluator. Server-side
  extraction and search may also use configured providers. Running the example
  can incur provider charges; no provider run is part of the offline tests.

The HTTP shapes are exercised by mocks. An actual server revision, its provider
configuration and its authorization/readiness behavior need separate
qualification before treating a dataset run as evidence.

## Generate hypotheses and grade

From the repository root, after configuring the isolated server, credentials
and dataset:

```bash
python loom/run_loom.py \
  --base-url http://127.0.0.1:7777 \
  --dataset data/longmemeval_s_cleaned.json \
  --out loom/loom_hyp.jsonl \
  --limit 40 --shuffle

python src/evaluation/evaluate_qa.py \
  gpt-4o loom/loom_hyp.jsonl data/longmemeval_s_cleaned.json
```

The limit is a sample, not a full benchmark. Omit it only for an intended full
dataset run with an appropriate provider budget. Namespace contents persist
for the optional later latency pass; use a dedicated test deployment and its
documented erasure procedure for your own fixtures after evaluation.

## Completion and accounting

The harness explicitly requests inline extraction. A deferred flag or nonempty
queue ID aborts the run before the affected question proceeds to retrieval or
its reader. Other concurrent questions may already have progressed. There is
no background extraction waiter or rollback of earlier work, and an aborted
run does not produce a completed scored population.

Handled HTTP/key failures retain empty hypotheses and false recall flags in a
normally completed run. Loaded-latency percentiles use only recorded samples,
with measured/skipped counts and null percentiles when no sample exists. A
reader failure can omit an earlier search sample; these are not all attempted
HTTP requests. The percentile index is `round((n - 1) * q)` over sorted samples.

`--measure-latency` repeats eligible searches sequentially after this harness's
ingestion. Existing output fields call these `latency_clean_*`; that label does
not establish server-wide quiescence or exclude other clients and workers.
It reports eligible, attempted, measured, skipped and failed logical-search
counts. A logical search can make multiple requests through the retry path.

Context estimates count rendered memory history plus structured operands at
four characters per token. Summary context estimates include all question
outcomes, with zero for handled failures lacking a context sample. These are
not provider usage, complete prompt sizes or cost measurements.

## Offline checks

Install the test-only dependency separately, then run the adapter test file:

```bash
python -m pip install pytest
python -m pytest loom/test_run_loom.py -q
```

HTTP, reader and question failures are mocked; this command requires no Loom
server, provider credentials, model calls or dataset download. The tests cover
response/retry behavior, reader request settings, dated rendering, extraction
completion and missing-latency/structured-context accounting.

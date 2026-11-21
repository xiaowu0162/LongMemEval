"""Run an external Loom memory service on LongMemEval histories.

Indexing and retrieval use Loom's HTTP API. The adapted facts/CoT reader adds
temporal precedence, counting/coreference, date arithmetic and optional
server-computed aggregate guidance. It is not an unchanged native reader.

Each question ingests its sessions in a separate preauthorized namespace,
retrieves memories and writes question_id/hypothesis JSONL for the unchanged
src/evaluation/evaluate_qa.py script. Custom retrieval diagnostics include all
question outcomes, unlike the native abstention-excluding retrieval metric.
The adapter does not emit native retrieval_results logs.

See loom/README.md for dependencies, identity prerequisites, provider usage and
validation limits. No benchmark score or live deployment is supplied here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import httpx

_HAYSTACK_DATE = re.compile(r"^(\d{4})/(\d{2})/(\d{2})")

# Adapted from the official facts/CoT template in src/generation/run_generation.py,
# with temporal precedence, counting/coreference, date arithmetic and optional
# server-computed aggregate guidance. A single user message, no system prompt;
# the full completion is the hypothesis the official evaluator grades.
_ANSWER_PROMPT = (
    "I will give you several facts extracted from history chats between you "
    "and a user. Please answer the question based on the relevant facts. "
    "Answer the question step by step: first extract all the relevant "
    "information, and then reason over the information to get the answer."
    "\n\nMemories are listed oldest-first by Date. When deriving the answer:"
    "\n- If two facts about the same attribute (a value, count, location, "
    "brand, goal, status) conflict, the MOST RECENT by Date is current — use "
    "it; do not average, sum, or call them contradictory."
    "\n- For count/sum/'how many'/'total' questions, enumerate every distinct "
    "qualifying instance as a list BEFORE counting; treat differing "
    "quantities, dates, or occasions as SEPARATE instances; merge facts that "
    "refer to the same person/thing (coreference); do NOT count "
    "planned/considered/hypothetical items as actual."
    "\n- For 'how long between'/'how many days' questions, use the Date of "
    "each relevant memory as the event date and compute the difference; only "
    "say you cannot compute it if a needed Date is genuinely absent."
    "\n- When a 'Computed from structured records' block is present below, it "
    "is an EXACT server-side aggregate (COUNT / SUM / date-diff) over "
    "per-occurrence records — treat it as authoritative for the numeric part "
    "of the answer and prefer it over re-counting the chats by hand, UNLESS "
    "the chats plainly show a qualifying instance it omitted."
    "{computed}"
    "\n\n\nHistory Chats:\n\n{history}\n\nCurrent Date: {date}\nQuestion: "
    "{question}\nAnswer (step by step):"
)


def _iso(longmemeval_date: str) -> str:
    """'2023/04/10 (Mon) 17:50' -> '2023-04-10' (the ISO prefix Loom's
    observation_date accepts); '' if unparseable."""
    m = _HAYSTACK_DATE.match(longmemeval_date or "")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


async def _post(client: httpx.AsyncClient, url: str, body: dict, token: str,
                *, retries: int = 3) -> dict:
    """POST with retry + backoff on transient failures: network/timeout errors,
    429, and 5xx. A dropped index/search would silently corrupt recall, so
    transient network blips and ClickHouse write contention must be ridden out.
    A non-429 4xx (a genuine client error) raises immediately, not retried."""
    if retries < 1:
        raise ValueError(f"retries must be >= 1, got {retries}")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    # Forward the optional organization header on reader requests only.
    _org = os.environ.get("OPENAI_ORGANIZATION")
    if _org and "api.openai.com" in url:
        headers["OpenAI-Organization"] = _org
    for attempt in range(retries):
        last_attempt = attempt == retries - 1
        try:
            r = await client.post(url, json=body, headers=headers, timeout=120.0)
        except httpx.RequestError:  # network / timeout — transient, retry
            if last_attempt:
                raise
            await asyncio.sleep(0.5 * (2 ** attempt))
            continue
        if 200 <= r.status_code < 300:
            return r.json()
        if 300 <= r.status_code < 400:
            # Explain how to select the final endpoint instead of following a
            # redirect for a request carrying credentials.
            raise RuntimeError(
                f"{url} returned {r.status_code} redirect to "
                f"{r.headers.get('location', '?')}; point --base-url at the "
                f"final URL (redirects are not followed on POST)."
            )
        if r.status_code != 429 and r.status_code < 500:
            r.raise_for_status()  # non-429 4xx: genuine client error, don't retry
        if last_attempt:
            r.raise_for_status()
        await asyncio.sleep(0.5 * (2 ** attempt))
    raise RuntimeError("unreachable")


def _pct(xs: list, q: float):
    """Select index round((n-1)*q) from sorted samples; return zero if empty."""
    return xs[round((len(xs) - 1) * q)] if xs else 0


def _history_block(hits: list[dict], key_to_date: dict | None = None) -> str:
    """Render retrieved memories as dated blocks, oldest first (mirrors the
    official run_generation.py per-session formatting)."""
    key_to_date = key_to_date or {}
    def date_of(h: dict) -> str:
        d = (h.get("valid_at") or h.get("temporal_anchor") or "").strip()
        if d.startswith("1970-01-01"):  # epoch sentinel = undated
            d = ""
        if not d:
            # Use the source-session date when the memory has no usable date.
            d = (key_to_date.get(str(h.get("memory_key") or "")) or "").strip()
        return d[:10] if d else ""

    blocks = []
    for i, h in enumerate(sorted(hits, key=lambda h: date_of(h) or "9999")):
        content = (h.get("content_excerpt") or "").strip()
        blocks.append(f"### Memory {i + 1}:\nDate: {date_of(h) or 'unknown'}\n"
                      f"Content:\n{content}\n")
    return "\n".join(blocks) or "(no facts retrieved)"


def _derived_block(derived: dict | None) -> str:
    """Render the server-computed derived aggregate as an
    authoritative operand block. Empty string when absent so the {computed}
    slot collapses on non-derived questions."""
    if not derived:
        return ""
    op = str(derived.get("op") or "")
    parts: list[str] = []
    cnt = derived.get("count")
    if cnt is not None:
        parts.append(f"occurrences counted: {cnt}")
    if derived.get("sum") is not None:
        parts.append(f"sum of amounts: {derived['sum']:g}")
    if derived.get("avg") is not None:
        parts.append(f"average amount: {derived['avg']:g}")
    fa, la = derived.get("first_at"), derived.get("last_at")
    if fa:
        parts.append(f"earliest occurrence: {str(fa)[:10]}")
    if la:
        parts.append(f"latest occurrence: {str(la)[:10]}")
    sd = derived.get("span_days")
    if sd is not None:
        parts.append(
            f"span first->last: {round(sd)} days (~{round(sd / 7)} weeks)"
        )
    items = derived.get("items") or []
    if items:
        listed = "; ".join(
            (it.get("object") or "").strip()
            + (
                f"={it['numeric_value']:g}{it.get('unit') or ''}"
                if it.get("numeric_value") is not None
                else ""
            )
            + (f" on {str(it.get('occurred_at'))[:10]}" if it.get("occurred_at") else "")
            for it in items[:40]
        )
        parts.append(f"instances: {listed}")
    if not parts:
        return ""
    hedge = (
        " (category filter was broadened to predicate-only — may include "
        "unrelated instances; cross-check the chats)"
        if derived.get("category_broadened")
        else ""
    )
    return (
        "\n\nComputed from structured records (exact aggregate over "
        f"per-occurrence rows; op={op}){hedge}:\n- " + "\n- ".join(parts)
    )


async def _answer(client: httpx.AsyncClient, question: str, hits: list[dict],
                  question_date: str, model: str, api_key: str,
                  key_to_date: dict | None = None,
                  derived: dict | None = None) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": _ANSWER_PROMPT.format(
            history=_history_block(hits, key_to_date),
            date=question_date or "unknown",
            question=question,
            computed=_derived_block(derived),
        )}],
    }
    # Reasoning models (gpt-5, o-series) reject temperature != 1 and use
    # max_completion_tokens (not max_tokens) — and a small cap truncates their
    # hidden reasoning. Non-reasoning temperature/output-cap settings match
    # run_generation.py (max_tokens = gen_length = 800 for the CoT prompt).
    # The reader prompt is adapted above; full reader parity is not claimed.
    # Reasoning models are left uncapped: the official harness predates them,
    # so there is no official cap to match.
    if not re.match(r"^(gpt-5|o[1-9])", model):
        body["temperature"] = 0.0
        body["max_tokens"] = 800
    # Route through _post so the reader call inherits the same retry/backoff as
    # the Loom calls — a transient 429/5xx/network error otherwise drops a whole
    # question and skews the metric.
    data = await _post(
        client, "https://api.openai.com/v1/chat/completions", body, api_key
    )
    return (data["choices"][0]["message"]["content"] or "").strip()


async def _run_item(client: httpx.AsyncClient, base_url: str, token: str, item: dict,
                    *, top_k: int, search_mode: str, ingest_conc: int,
                    model: str, api_key: str, retrieval_budget: str,
                    item_sem: asyncio.Semaphore) -> dict:
    async with item_sem:
        ns = f"lme-{uuid.uuid4().hex[:10]}"
        identity = {"org": "dev", "namespace": ns, "agent": "lme-loom", "user_id": "-"}
        sessions = item.get("haystack_sessions", []) or []
        dates = item.get("haystack_dates", []) or []
        sids = item.get("haystack_session_ids", []) or []
        key_to_sessions: dict[str, set[str]] = {}
        # session_id -> ISO observation date (the bench holds these; used as the
        # date-fallback when a memory's temporal_anchor/valid_at is NULL).
        sid_to_date: dict[str, str] = {
            str(sids[i]): _iso(dates[i])
            for i in range(min(len(sids), len(dates)))
            if sids[i] and _iso(dates[i])
        }

        # 1-2) Index each session with bounded concurrency and server-side extraction.
        ingest_sem = asyncio.Semaphore(max(1, ingest_conc))

        async def ingest(i: int, session: list) -> None:
            if not session:
                return
            sid = str(sids[i]) if i < len(sids) and sids[i] else ""
            body: dict = {**identity, "messages": session, "max_tokens": 4096,
                          "defer": False}
            if sid:
                body["session_id"] = sid
            if i < len(dates) and _iso(dates[i]):
                body["observation_date"] = _iso(dates[i])
            async with ingest_sem:
                resp = await _post(client, base_url + "/v1/memory.set_from_messages", body, token)
            if resp.get("deferred") or resp.get("queue_id"):
                raise RuntimeError("Deferred extraction cannot be searched before completion")
            for w in resp.get("written") or []:
                key = str(w.get("memory_key") or "")
                if key:
                    key_to_sessions.setdefault(key, set()).add(sid)

        await asyncio.gather(*(ingest(i, s) for i, s in enumerate(sessions)))

        # 3) Retrieve using the selected search mode.
        search_body: dict = {
            **identity, "namespace_scope": "this",
            "query": str(item["question"]), "top_k": top_k,
            "search_mode": search_mode, "alpha": 0.5, "include_top_n_unmatched": 120,
        }
        # Forward an explicitly selected retrieval budget.
        if retrieval_budget:
            search_body["retrieval_budget"] = retrieval_budget
        q_iso = _iso(str(item.get("question_date", "")))
        if q_iso:
            search_body["observation_date"] = q_iso
        _t0 = time.perf_counter()
        resp = await _post(client, base_url + "/v1/memory.search", search_body, token)
        search_ms = (time.perf_counter() - _t0) * 1000.0  # NB: under-load (concurrent ingest)
        hits = resp.get("results", [])
        hyde_fired = bool(resp.get("hyde_fallback_used"))
        # Server-computed COUNT/SUM/date-diff over per-occurrence
        # derived_facts. None on non-derived questions / empty match — the
        # reader then falls back to counting the recalled passages by hand.
        derived = resp.get("derived_aggregate")

        # Custom diagnostic: any labelled evidence session in retrieved memory.
        # Aggregation below includes all questions, including abstention cases.
        answer_sessions = {str(s) for s in (item.get("answer_session_ids") or []) if s}
        retrieved = set()
        for h in hits[:top_k]:
            retrieved |= key_to_sessions.get(str(h.get("memory_key") or ""), set())
        retrieved.discard("")
        recalled = bool(answer_sessions & retrieved) if answer_sessions else False
        # ALL-session coverage@k: did the top-k cover EVERY gold evidence session?
        # Any-session coverage alone does not establish complete multi-hop evidence.
        all_covered = bool(answer_sessions) and answer_sessions <= retrieved

        # Diagnostic answer-string match in retrieved excerpts, not a QA judge.
        gold = re.sub(r"\s+", " ", str(item.get("answer", "")).strip().lower())

        def _present(hay: str) -> bool:
            # Word-boundary match for short golds ("4", "nike") so they don't
            # spuriously match inside other tokens; substring for long ones.
            if not gold:
                return False
            if len(gold) <= 12:
                return re.search(r"(?<![a-z0-9])" + re.escape(gold) + r"(?![a-z0-9])", hay) is not None
            return gold in hay

        fact_in_context = any(
            _present(re.sub(r"\s+", " ", (h.get("content_excerpt") or "").lower()))
            for h in hits[:top_k]
        )

        # 4) Read: generate an answer with the adapted reader prompt.
        # memory_key -> source-session date (earliest), for the date-fallback.
        key_to_date = {
            k: min((sid_to_date[s] for s in ss if s in sid_to_date), default="")
            for k, ss in key_to_sessions.items()
        }
        # Estimate retrieval context only, not the full prompt or billed tokens.
        ctx_tokens = (len(_history_block(hits, key_to_date))
                      + len(_derived_block(derived))) // 4
        hypothesis = await _answer(client, str(item["question"]), hits,
                                   str(item.get("question_date", "")), model, api_key,
                                   key_to_date=key_to_date, derived=derived)
        return {
            "question_id": str(item["question_id"]),
            "question_type": str(item.get("question_type", "")),
            "hypothesis": hypothesis,
            "recalled": recalled,
            "all_covered": all_covered,
            "fact_in_context": fact_in_context,
            "ctx_tokens": ctx_tokens,
            "search_ms_loaded": round(search_ms, 1),
            "hyde_fired": hyde_fired,
            "n_hits": len(hits),
            "ns": ns,
            "query": str(item["question"]),
            "q_iso": q_iso,
        }


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:7777", help="Loom server URL")
    p.add_argument("--token", default=os.environ.get("LOOM_TOKEN", ""), help="Loom bearer token")
    p.add_argument("--dataset", default="data/longmemeval_s_cleaned.json")
    p.add_argument("--out", default="loom/loom_hyp.jsonl", help="hypotheses JSONL for evaluate_qa.py")
    p.add_argument("--limit", type=int, default=0, help="cap questions (0 = all 500)")
    p.add_argument("--shuffle", action="store_true",
                   help="shuffle before --limit (the dataset is category-ordered, so a "
                        "bare --limit samples a single question type). Deterministic via --seed.")
    p.add_argument("--seed", type=int, default=42, help="shuffle seed")
    p.add_argument("--question-type", default="",
                   help="comma-separated question_type filter (e.g. "
                        "single-session-assistant,knowledge-update); applied before "
                        "--shuffle/--limit so a category can be run complete. Empty = all.")
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--search-mode", default="rrf",
                   help="Loom search mode; 'rrf' = let Loom's planner self-route")
    p.add_argument("--concurrency", type=int, default=4, help="questions in flight")
    p.add_argument("--ingest-concurrency", type=int, default=8,
                   help="concurrent index calls per question")
    p.add_argument("--answer-model", default="gpt-4o", help="reader model (OpenAI)")
    p.add_argument("--retrieval-budget", default="",
                   help="optional Loom retrieval budget; omitted when empty")
    p.add_argument("--measure-latency", action="store_true",
                   help="repeat eligible searches sequentially after this run's ingestion; "
                        "does not establish server-wide quiescence")
    p.add_argument("--metrics-out", default="",
                   help="write the latency/token/recall/HyDE metrics summary as JSON here")
    args = p.parse_args()
    if args.concurrency < 1:
        p.error("--concurrency must be at least 1")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("OPENAI_API_KEY is required (reader model).", file=sys.stderr)
        return 2
    ds_path = Path(args.dataset)
    if not ds_path.exists():
        print(f"missing dataset: {ds_path}\n  wget https://huggingface.co/datasets/"
              "xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json "
              f"-O {ds_path}", file=sys.stderr)
        return 2
    dataset = json.loads(ds_path.read_text())
    if args.question_type:
        wanted = {t.strip() for t in args.question_type.split(",") if t.strip()}
        dataset = [d for d in dataset if str(d.get("question_type", "")) in wanted]
    if args.shuffle:
        random.Random(args.seed).shuffle(dataset)
    if args.limit > 0:
        dataset = dataset[: args.limit]
    print(f"loom-longmemeval: {len(dataset)} questions, top_k={args.top_k}, "
          f"search_mode={args.search_mode}, model={args.answer_model}", flush=True)

    item_sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        h = await client.get(args.base_url + "/v1/health", timeout=5.0)
        if h.status_code != 200:
            print(f"Loom server unhealthy: {h.status_code}", file=sys.stderr)
            return 2

        async def runner(item: dict) -> None:
            try:
                r = await _run_item(client, args.base_url, args.token, item,
                                    top_k=args.top_k, search_mode=args.search_mode,
                                    ingest_conc=args.ingest_concurrency,
                                    model=args.answer_model, api_key=api_key,
                                    retrieval_budget=args.retrieval_budget,
                                    item_sem=item_sem)
                results.append(r)
                print(f"  {'✓' if r['recalled'] else '✗'} {r['question_id']} "
                      f"[{r['question_type']}]", flush=True)
            except (httpx.HTTPError, KeyError) as e:
                print(f"  ! {item.get('question_id', '?')} ERROR: {e}", file=sys.stderr, flush=True)
                # Record a placeholder so a harness failure still counts in the
                # denominator (empty hypothesis -> judged wrong) rather than
                # silently dropping the question and inflating QA accuracy/recall.
                results.append({
                    "question_id": item.get("question_id", ""),
                    "question_type": item.get("question_type", "unknown"),
                    "hypothesis": "",
                    "recalled": False,
                    "all_covered": False,
                })

        await asyncio.gather(*(runner(it) for it in dataset))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps({"question_id": r["question_id"], "hypothesis": r["hypothesis"]}) + "\n")

    # Retrieval recall@k by question type (Loom's own metric; the QA score comes
    # from evaluate_qa.py on the hypotheses file).
    by_type: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r["recalled"])
    print(f"\nrecall@{args.top_k} by question_type:")
    for qt in sorted(by_type):
        v = by_type[qt]
        print(f"  {qt:28} {sum(v)}/{len(v)} ({sum(v) / len(v) * 100:.1f}%)")
    tot = [r["recalled"] for r in results]
    print(f"  {'OVERALL':28} {sum(tot)}/{len(tot)} "
          f"({sum(tot) / len(tot) * 100:.1f}%)" if tot else "  (no results)")

    # ALL-session coverage@k: did the top-k cover EVERY gold evidence session?
    # (multi-hop completeness — the ANY-session recall above hides this)
    cby: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        cby[r["question_type"]].append(r.get("all_covered", False))
    print(f"\nALL-session coverage@{args.top_k} (every gold session in top-k — multi-hop completeness):")
    for qt in sorted(cby):
        v = cby[qt]
        print(f"  {qt:28} {sum(v)}/{len(v)} ({sum(v) / len(v) * 100:.1f}%)")
    ctot = [r.get("all_covered", False) for r in results]
    print(f"  {'OVERALL':28} {sum(ctot)}/{len(ctot)} "
          f"({sum(ctot) / len(ctot) * 100:.1f}%)" if ctot else "  (no results)")

    # Fact-level recall@k by question type: did the gold answer string actually
    # reach the reader? This string diagnostic is separate from judged QA.
    fby: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        fby[r["question_type"]].append(r.get("fact_in_context", False))
    print(f"\nFACT-level recall@{args.top_k} (gold answer present in a retrieved excerpt):")
    for qt in sorted(fby):
        v = fby[qt]
        print(f"  {qt:28} {sum(v)}/{len(v)} ({sum(v) / len(v) * 100:.1f}%)")
    ftot = [r.get("fact_in_context", False) for r in results]
    print(f"  {'OVERALL':28} {sum(ftot)}/{len(ftot)} "
          f"({sum(ftot) / len(ftot) * 100:.1f}%)" if ftot else "  (no results)")

    # Token efficiency: the size of the context Loom hands the reader per query.
    toks = sorted(r.get("ctx_tokens", 0) for r in results)
    tok_median = _pct(toks, 0.5)
    tok_mean = round(sum(toks) / len(toks)) if toks else 0
    mem_mean = sum(r.get("n_hits", 0) for r in results) // max(1, len(results))
    print(f"\nTOKEN efficiency (context served to reader, ~4 chars/token):"
          f"\n  median {tok_median} tok/query   mean {tok_mean}   (~{mem_mean} memories/query)")

    # HyDE fallback firing rate (recall-rescue LLM call; fires only on a weak top hit).
    hyde_n = sum(1 for r in results if r.get("hyde_fired"))
    print(f"\nHyDE fallback fired on {hyde_n}/{len(results)} "
          f"({hyde_n / len(results) * 100:.1f}%) queries" if results else "")

    # In-run search latency is measured UNDER concurrent-ingest load, which
    # inflates it — reported separately from the clean number below.
    ld = sorted(r["search_ms_loaded"] for r in results
                if r.get("search_ms_loaded") is not None)
    if ld:
        print(f"\nIn-run search latency UNDER LOAD (concurrent ingest — not comparable): "
              f"p50 {_pct(ld, 0.5):.0f}ms  p95 {_pct(ld, 0.95):.0f}ms")
    else:
        print("\nIn-run search latency: no completed search samples")
    print(f"In-run search latency counts: measured {len(ld)}, "
          f"skipped (no sample) {len(results) - len(ld)}")

    # Re-search eligible questions sequentially after this harness's ingestion.
    # Other clients or background work may still load the server.
    # Namespaces persist after the run.
    # Failed-question placeholders remain in accuracy/recall denominators but
    # have no retrieval identity to re-search for latency.
    latency_results = [r for r in results if r.get("ns") and r.get("query")]
    clean: list[float] = []
    clean_attempted = 0
    if args.measure_latency and latency_results:
        print(f"\nmeasuring CLEAN serving latency over {len(latency_results)} eligible sequential post-ingestion searches...", flush=True)
        async with httpx.AsyncClient(timeout=120.0) as lc:
            for r in latency_results:
                sb = {"org": "dev", "namespace": r["ns"], "agent": "lme-loom", "user_id": "-",
                      "namespace_scope": "this", "query": r["query"], "top_k": args.top_k, "search_mode": args.search_mode,
                      "alpha": 0.5, "include_top_n_unmatched": 120}
                if r.get("q_iso"):
                    sb["observation_date"] = r["q_iso"]
                if args.retrieval_budget:  # measure the same path the run used
                    sb["retrieval_budget"] = args.retrieval_budget
                clean_attempted += 1
                t0 = time.perf_counter()
                try:
                    await _post(lc, args.base_url + "/v1/memory.search", sb, args.token)
                except httpx.HTTPError:
                    continue
                clean.append((time.perf_counter() - t0) * 1000.0)
        clean.sort()
        if clean:
            print(f"CLEAN serving latency (post-ingestion, 1 query at a time): "
                  f"p50 {_pct(clean, 0.5):.0f}ms  p95 {_pct(clean, 0.95):.0f}ms  min {clean[0]:.0f}ms")
    if args.measure_latency:
        print(f"CLEAN serving latency counts: eligible {len(latency_results)}, "
              f"attempted {clean_attempted}, measured {len(clean)}, "
              f"skipped (no retrieval identity) {len(results) - len(latency_results)}, "
              f"failed {clean_attempted - len(clean)}")

    if args.metrics_out and results:
        n = len(results)
        metrics = {
            "n_questions": n, "top_k": args.top_k, "answer_model": args.answer_model,
            "recall_session_pct": round(sum(r["recalled"] for r in results) / n * 100, 1),
            "recall_allsession_pct": round(sum(r.get("all_covered", False) for r in results) / n * 100, 1),
            "recall_fact_pct": round(sum(r.get("fact_in_context", False) for r in results) / n * 100, 1),
            "ctx_tokens_median": tok_median, "ctx_tokens_mean": tok_mean,
            "hyde_fired_pct": round(hyde_n / n * 100, 1),
            "latency_loaded_p50_ms": round(_pct(ld, 0.5)) if ld else None,
            "latency_loaded_p95_ms": round(_pct(ld, 0.95)) if ld else None,
            "latency_loaded_measured_n": len(ld),
            "latency_loaded_skipped_n": n - len(ld),
            "latency_clean_p50_ms": round(_pct(clean, 0.5)) if clean else None,
            "latency_clean_p95_ms": round(_pct(clean, 0.95)) if clean else None,
            "latency_clean_eligible_n": len(latency_results) if args.measure_latency else None,
            "latency_clean_attempted_n": clean_attempted if args.measure_latency else None,
            "latency_clean_measured_n": len(clean) if args.measure_latency else None,
            "latency_clean_skipped_n": n - len(latency_results) if args.measure_latency else None,
            "latency_clean_failed_n": clean_attempted - len(clean) if args.measure_latency else None,
        }
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2))
        print(f"\nwrote metrics -> {args.metrics_out}")

    print(f"\nwrote {out_path}\nNow grade with the official evaluator:\n"
          f"  python src/evaluation/evaluate_qa.py gpt-4o {out_path} {args.dataset}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

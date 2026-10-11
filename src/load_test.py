"""Repeatable HTTP load test for the /predict endpoint.

Repeatability comes from: a fixed request count, fixed concurrency, a seeded
choice of payloads from the held-out test split, a warm-up phase excluded from
the measurement, and multiple runs whose spread is reported.

Usage:
    python -m src.load_test --url http://127.0.0.1:8000 --label local
    python -m src.load_test --url http://<ec2-ip>:8000 --label ec2 --runs 3
"""
import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def build_payloads(count, seed):
    data = load_breast_cancer()
    _, X_test, _, _ = train_test_split(
        data.data,
        data.target,
        test_size=0.20,
        random_state=42,
        stratify=data.target,
    )
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, len(X_test), size=count)
    return [{"features": X_test[i].tolist()} for i in rows]


async def run_once(url, payloads, concurrency, timeout):
    queue = asyncio.Queue()
    for payload in payloads:
        queue.put_nowait(payload)

    latencies = []
    errors = 0

    async def worker(client):
        nonlocal errors
        while True:
            try:
                payload = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            start = time.perf_counter()
            try:
                response = await client.post("/predict", json=payload)
                ok = response.status_code == 200
            except httpx.HTTPError:
                ok = False
            elapsed_ms = (time.perf_counter() - start) * 1000
            if ok:
                latencies.append(elapsed_ms)
            else:
                errors += 1

    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(base_url=url, timeout=timeout, limits=limits) as client:
        started = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
        duration = time.perf_counter() - started

    total = len(payloads)
    return {
        "requests": total,
        "errors": errors,
        "error_rate": errors / total,
        "duration_s": duration,
        "throughput_rps": total / duration,
        "p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
        "p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "p99_ms": float(np.percentile(latencies, 99)) if latencies else None,
        "max_ms": float(np.max(latencies)) if latencies else None,
    }


async def main_async(args):
    payloads = build_payloads(args.warmup + args.requests, args.seed)
    warmup, measured = payloads[:args.warmup], payloads[args.warmup:]

    async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout) as client:
        health = await client.get("/health")
        health.raise_for_status()

    if warmup:
        await run_once(args.url, warmup, args.concurrency, args.timeout)

    runs = []
    for run in range(1, args.runs + 1):
        result = await run_once(args.url, measured, args.concurrency, args.timeout)
        result["run"] = run
        runs.append(result)
        print(
            f"run {run}: {result['throughput_rps']:.1f} req/s  "
            f"p50={result['p50_ms']:.2f}ms  p95={result['p95_ms']:.2f}ms  "
            f"p99={result['p99_ms']:.2f}ms  errors={result['errors']}"
        )
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--label", default="local", help="name used in the report file")
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-error-rate", type=float, default=0.0,
                        help="exit non-zero if any run exceeds this error rate")
    args = parser.parse_args()

    runs = asyncio.run(main_async(args))

    p95s = [r["p95_ms"] for r in runs if r["p95_ms"] is not None]
    summary = {
        "label": args.label,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "url": args.url,
        "config": {
            "requests": args.requests,
            "concurrency": args.concurrency,
            "warmup": args.warmup,
            "runs": args.runs,
            "seed": args.seed,
        },
        "runs": runs,
        "mean_throughput_rps": statistics.mean(r["throughput_rps"] for r in runs),
        "mean_p95_ms": statistics.mean(p95s) if p95s else None,
        "p95_stdev_ms": statistics.stdev(p95s) if len(p95s) > 1 else 0.0,
    }

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"load_test_{args.label}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Append one row per invocation so runs can be compared over time.
    history = REPORTS / "load_test_history.csv"
    row = pd.DataFrame([{
        "timestamp": summary["timestamp"],
        "label": args.label,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "runs": args.runs,
        "mean_throughput_rps": summary["mean_throughput_rps"],
        "mean_p95_ms": summary["mean_p95_ms"],
        "p95_stdev_ms": summary["p95_stdev_ms"],
        "max_error_rate": max(r["error_rate"] for r in runs),
    }])
    row.to_csv(history, mode="a", header=not history.exists(), index=False)

    print(
        f"\nmean throughput {summary['mean_throughput_rps']:.1f} req/s, "
        f"mean p95 {summary['mean_p95_ms']:.2f}ms "
        f"(stdev {summary['p95_stdev_ms']:.2f}ms across {args.runs} runs)"
    )
    print(f"Saved report: {REPORTS / f'load_test_{args.label}.json'}")

    if max(r["error_rate"] for r in runs) > args.max_error_rate:
        raise SystemExit("error rate above threshold")


if __name__ == "__main__":
    main()

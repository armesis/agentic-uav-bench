#!/usr/bin/env python3
"""Batch runner. Resumable: finished runs (summary.json without harness_error) are skipped.

Examples
  # harness check, no LLM (expect ~100% success on B and C):
  python run.py --provider scripted --model oracle --conditions B,C --reps 1

  # one quick LLM pilot run:
  python run.py --provider openai --model gpt-5.6-luna --conditions C --missions M1 --reps 1

  # full batch (3 x 4 x 10 = 120 runs), randomized order:
  python run.py --provider openai --model gpt-5.6-luna --reps 10
"""
import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

from bench.agent import run_one
from bench.config import CFG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True,
                    choices=["openai", "anthropic", "ollama", "gemini", "openrouter", "scripted"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--conditions", default="A,B,C")
    ap.add_argument("--missions", default="M1,M2,M3,M4")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--rep-start", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="use -1 to omit (provider default; needed for some reasoning models)")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--reasoning-effort", default=None,
                    help="none|low|medium|high; default: 'none' for gpt-5.x/6 (required for tools on chat API)")
    ap.add_argument("--budget-usd", type=float, default=4.0,
                    help="stop the batch when spend for this model (all runs in --out) reaches this")
    ap.add_argument("--px4-dir", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--retry-errors", action="store_true", help="re-run runs that ended in harness_error")
    args = ap.parse_args()

    if args.px4_dir:
        CFG.sim.px4_dir = args.px4_dir
    temp = None if args.temperature < 0 else args.temperature
    conds = args.conditions.split(",")
    missions = args.missions.split(",")
    jobs = [(c, m, r) for r in range(args.rep_start, args.rep_start + args.reps)
            for c in conds for m in missions]
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(jobs)   # interleave to avoid time-of-day API drift confounds

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.model.replace("/", "_").replace(":", "_")
    def spent():
        tot = 0.0
        for p in out.glob(f"{tag}__*/summary.json"):
            tot += json.loads(p.read_text()).get("cost_usd") or 0.0
        return tot

    t0 = time.time()
    done_now = 0
    for i, (c, m, r) in enumerate(jobs, 1):
        so_far = spent()
        if so_far >= args.budget_usd:
            print(f"BUDGET STOP: ${so_far:.3f} spent >= --budget-usd {args.budget_usd}. Re-run to resume.")
            break
        sp = out / f"{tag}__{c}__{m}__r{r:02d}" / "summary.json"
        if sp.exists():
            s = json.loads(sp.read_text())
            if s.get("termination") != "harness_error" or not args.retry_errors:
                print(f"[{i}/{len(jobs)}] skip {sp.parent.name} ({s.get('termination')})")
                continue
        s = asyncio.run(run_one(CFG, c, m, args.provider, args.model, r, out,
                                temperature=temp, base_url=args.base_url,
                                reasoning_effort=args.reasoning_effort))
        done_now += 1
        el = (time.time() - t0) / 60
        cost = spent()
        print(f"[{i}/{len(jobs)}] {s['run_id']}: success={s.get('success')} "
              f"term={s.get('termination')} fence={s.get('fence_breach')} crash={s.get('crash')} "
              f"llm_calls={s.get('n_llm_calls')} wall={s.get('run_wall_s')}s "
              f"cost=${s.get('cost_usd') or 0:.4f} total=${cost:.3f}  (elapsed {el:.1f} min)")
        if s.get("termination") == "harness_error":
            print("   ERROR:", s.get("error"), file=sys.stderr)
        if s.get("termination") == "api_fatal":
            print("   FATAL API ERROR (quota/key/model) - stopping:", s.get("error"), file=sys.stderr)
            (out / f"{tag}__{c}__{m}__r{r:02d}" / "summary.json").unlink(missing_ok=True)
            break


if __name__ == "__main__":
    main()

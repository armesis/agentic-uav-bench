# Runbook: running the full experiment

Step-by-step manual for running the benchmark on this machine (Ubuntu 24.04, Gazebo Harmonic 8.15,
RTX 3050 6 GB), plus every problem hit during bring-up on 2026-09-28 and how it was fixed.
`README.md` has the experimental design; this file is the "how do I actually run it" companion.

**Time budget.** Oracle runs take ~1.5–2 min each (sim boot ≈ 15–20 s + flight). LLM runs take 1–7 min
(the agent phase is capped at 420 s). A full 120-run batch is roughly **4–12 h per model**, depending on
how long the agent flies. Everything is resumable, so a batch can be split across sessions.

---

## 0. One-time setup (already done on this machine)

1. **PX4.** `PX4-Autopilot/` in this repo is a git worktree of the shared store
   `~/.local/share/px4/PX4-Autopilot.git`, pinned to **stock v1.17.0** (branch `agentic-uav-bench`).
   It is gitignored. Do not patch it: the paper claims stock PX4. See `CLAUDE.md` and
   `~/.local/share/px4/README.md`.
   Rebuild (after a `git clean` or a submodule change):
   ```bash
   cd ~/agentic-uav-bench/PX4-Autopilot && make px4_sitl
   ```
   `bench/config.py` finds it automatically; `PX4_DIR=...` overrides.
2. **Python env.**
   ```bash
   cd ~/agentic-uav-bench && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```
   Always call `.venv/bin/python` (or `source .venv/bin/activate` first).

## 1. Before every session

```bash
cd ~/agentic-uav-bench
pkill -9 -f px4_sitl_default/bin/px4; pkill -9 -f "gz sim"; pkill -9 -f mavsdk_server   # leftovers
df -h /          # keep a few GB free: every run writes a ~25 MB PX4 ulog
```

## 2. Offline self-test (no simulator, ~2 min)

```bash
.venv/bin/python -m tests.test_offline        # must end with: ALL PASSED
```

## 3. Harness check with the oracle (no LLM, ~13 min)

The oracle is a hand-written correct pilot. It must pass **8/8** before any LLM run; otherwise the
harness, not the model, is being measured.

```bash
.venv/bin/python run.py --provider scripted --model oracle --conditions B,C --reps 1 --out smoke
```

Watch it fly (opens the Gazebo window; good for debugging a single run):

```bash
HEADLESS=0 .venv/bin/python run.py --provider scripted --model oracle --conditions C --missions M1 --reps 1 --out smoke_gui
```

Sanity check that every takeoff went to the requested altitude (not 2.5 m):

```bash
grep -h 'takeoff altitude' smoke/*/px4.log
```

## 4. Local model via LM Studio

1. Load the model **from the CLI with an explicit context length**. If you let LM Studio auto-load it on
   the first request, it uses the default 4096 context, and a mission overflows it.
   ```bash
   ~/.lmstudio/bin/lms load nvidia/nemotron-3-nano-4b --context-length 16384 --gpu max -y
   ~/.lmstudio/bin/lms ps          # check CONTEXT = 16384
   ```
2. **Decide the reasoning setting before the batch and do not touch it while runs are going.** Toggling it
   mid-run mixes two configurations in one dataset (that happened once; those runs were set aside in
   `pilot_nemotron_mixed/`). Nemotron with reasoning off answers in ~0.5 s, with it on in ~2–12 s.
3. Quick check that tool calling works and reasoning is really off (`reasoning_tokens=0`):
   ```bash
   curl -s localhost:1234/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"nvidia/nemotron-3-nano-4b","messages":[{"role":"user","content":"Arm the drone."}],"tools":[{"type":"function","function":{"name":"arm","description":"Arm the motors","parameters":{"type":"object","properties":{}}}}]}' | python3 -m json.tool | grep -E '"name"|reasoning_tokens'
   ```
4. Pilot (1 rep per condition on M1 and M2, ~30 min), then read transcripts (step 7):
   ```bash
   .venv/bin/python run.py --provider lmstudio --model nvidia/nemotron-3-nano-4b --conditions A,B,C --missions M1,M2 --reps 1 --out pilot_nemotron --model-notes "reasoning off, Q4_K_M, ctx 16384, RTX 3050 6GB"
   ```
5. Full batch (120 runs):
   ```bash
   .venv/bin/python run.py --provider lmstudio --model nvidia/nemotron-3-nano-4b --reps 10 --out results --model-notes "reasoning off, Q4_K_M, ctx 16384, RTX 3050 6GB"
   ```
   Always pass the same `--model-notes`: it is stored in every `summary.json`, and the paper has to report
   quantization, context, reasoning setting and GPU.

## 5. Cloud model (OpenAI)

```bash
export OPENAI_API_KEY=...        # in the same terminal that launches the run
.venv/bin/python run.py --provider openai --model gpt-5.6-luna --missions M1 --reps 1 --out pilot_gpt     # pilot, 3 runs
.venv/bin/python run.py --provider openai --model gpt-5.6-luna --reps 10 --out results                    # full batch
```

`reasoning_effort=none` is sent automatically for gpt-5.x. The batch stops at `--budget-usd` (default 4.0)
and on quota/auth errors.

## 6. Running two models at once (optional)

Each batch needs its own `--instance` (separate PX4 instance, ports and Gazebo partition):

```bash
.venv/bin/python run.py --instance 0 --provider openai   --model gpt-5.6-luna              --reps 10 --out results
.venv/bin/python run.py --instance 1 --provider lmstudio --model nvidia/nemotron-3-nano-4b --reps 10 --out results --model-notes "..."
```

**Not validated yet.** Before relying on it, run two oracle sweeps at the same time (`--out smoke0` /
`--instance 1 --out smoke1`) and check that both pass and the sim stays real-time. If they don't, run the
models one after the other.

## 7. Stopping, resuming, inspecting

- **Stop:** Ctrl-C in the terminal. Then clean up leftovers (step 1). The interrupted run has no
  `summary.json` and is simply re-run next time.
- **Resume:** re-run the exact same command. Finished runs are skipped; add `--retry-errors` to redo
  runs that ended in `harness_error`.
- **Per-run output** is in `<out>/<model>__<cond>__<mission>__rNN/`:
  `summary.json` (metrics), `events.jsonl` (every LLM call and command), `transcript.json`,
  `telemetry.csv` (5 Hz), `px4.log`. `events.jsonl` and `telemetry.csv` are written **at the end** of a
  run, so while a run is going, follow `px4.log` instead:
  ```bash
  tail -f "$(ls -td results/*/ | head -1)px4.log" | grep -E 'navigator|commander'
  ```
- Quick table of finished runs:
  ```bash
  for f in results/*/summary.json; do python3 -c "import json;s=json.load(open('$f'));print(s['run_id'],s['termination'],s['success'],s.get('n_llm_calls'))"; done
  ```

## 8. Analysis

```bash
.venv/bin/python analyze.py --results results --out paper_out
```

Rules (see `CLAUDE.md`): harness errors are excluded from the stats, agent failures (timeouts,
context overflow) are not. Don't change missions, metrics or prompts after the full batch starts; if you
have to, bump a version tag and re-run everything.

---

## Troubleshooting: problems hit so far and their fixes

| Symptom | Cause | Fix |
|---|---|---|
| `px4.log`: `symbol lookup error: ... libgz-sim-*.so: undefined symbol ... gz::transport::v13::Node::Subscriber` | An old source-built gz-transport 13.4.1 in `/usr/local/lib` shadows apt's 13.6, which gz-sim 8.15 needs. | Fixed in `bench/sim.py` (puts `/usr/lib/x86_64-linux-gnu` first in `LD_LIBRARY_PATH` for PX4/Gazebo). A permanent system fix is to remove the `/usr/local` gz-transport, but check nothing else needs it first. |
| `summary.json`: `harness_error`, `TimeoutError` inside `mavsdk ... channel_ready` | Not a MAVSDK bug: `mavsdk_server` doesn't open gRPC until it sees a vehicle, so this means **PX4/Gazebo never came up**. | Read that run's `px4.log`; the real error is there. |
| `px4.log`: `Unable to find or download file`, then `Waiting for Gazebo world...` / `Timed out waiting for Gazebo world` | The `Tools/simulation/gz` models submodule is empty. | `cd PX4-Autopilot && git submodule update --init Tools/simulation/gz` (check `ls Tools/simulation/gz/worlds/default.sdf`). |
| Every takeoff goes to **2.5 m** (`grep 'takeoff altitude' */px4.log`), then everything times out | A takeoff sent right after the arm ACK, before PX4 has fully armed, makes PX4 ignore the requested altitude. | Fixed in `bench/vehicle.py`: `takeoff()` waits for the armed flag first. Re-check with the grep in step 3. |
| Condition C: `takeoff` rejected with `NOT_ARMED` right after a successful `arm` | The arm ACK arrives before the armed flag reaches telemetry. | Fixed: C's `arm` now blocks until armed. |
| Oracle flies perfectly but `success=False`, `fence_breach=True` | PX4's RTL climbs to 30 m (default) > 25 m ceiling. | Fixed: harness sets `RTL_RETURN_ALT=15`, `RTL_DESCEND_ALT=10`. |
| `HEADLESS=0` still headless | PX4 treats *any* non-empty `HEADLESS` as headless. | Fixed: the harness unsets it when `HEADLESS=0`. |
| `Received ack for not-existing command: 176! Ignoring...` in the terminal | MAVSDK noise (a duplicate `DO_SET_MODE` ACK). | Harmless; ignore. |
| LM Studio run ends with `termination=context_overflow` | Model loaded with the default 4096 context. | Load with `lms load ... --context-length 16384` (step 4). |
| Local-model latencies jump around mid-batch | Reasoning toggled, or model reloaded with different settings, during the batch. | Don't touch LM Studio while a batch runs; note settings in `--model-notes`. |
| One run hung at ~0.5 m with all commands timing out (seen once, before the fixes above) | Not reproduced since (32 later runs were fine). | If it comes back: keep the run folder, check `px4.log` and `telemetry.csv`, and re-run with `--retry-errors`. |
| `pip`: `cannot execute: required file not found` | Scripts left over from removed Python 3.9–3.11 installs. | Already cleaned; use `.venv/bin/python -m pip` or `python3 -m pip`. |
| `git submodule update` hangs cloning `PX4-gazebo-models` | Slow GitHub download of a large repo. | Clone from a local copy: `git -c protocol.file.allow=always submodule update ...` with the URL pointed at px4_ws's module (see `~/.local/share/px4/README.md`). |
| Disk full | ~25 MB PX4 ulog per run, plus caches. | `df -h /`; old `smoke*/` and `pilot_*/` folders can be deleted. |

## Qualitative behaviours worth noting (pilot, Nemotron-3-Nano-4B)

- **A (free-form):** writes the whole plan, including `mission_complete`, as one reply. All commands run
  within ~0.1 s and overwrite each other, so the vehicle never flies the mission. It also commands the
  out-of-fence P2 on M2.
- **B (typed tools):** skipped `arm`, got "accepted" ACKs for everything, ignored `armed: false` in the
  returned state, and reported a successful mission while sitting on the ground (hallucinated success).
- **C (guard):** the guard's `NOT_ARMED` / `OUTSIDE_ENVELOPE` rejections are what catch these.

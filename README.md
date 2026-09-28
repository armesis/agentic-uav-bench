# agentic_uav_bench

Controlled PX4 SITL study: **does the agent–autopilot interface design change how well an LLM agent flies a UAV mission?**

## Setup (Ubuntu 22.04/24.04)

```bash
# 1. PX4 (skip if already built)
git clone https://github.com/PX4/PX4-Autopilot.git --recursive ~/PX4-Autopilot
bash ~/PX4-Autopilot/Tools/setup/ubuntu.sh
cd ~/PX4-Autopilot && make px4_sitl gz_x500        # builds + opens sim once; Ctrl-C after it flies

# 2. Python deps
cd agentic_uav_bench
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. key(s)
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY
export PX4_DIR=~/PX4-Autopilot   # if not at ~/PX4-Autopilot
```

## Run order

```bash
# 0) offline self-test, no PX4 needed (should print ALL PASSED)
python -m tests.test_offline

# 1) harness check against real SITL, no LLM (expect success=True)
python run.py --provider scripted --model oracle --conditions C --missions M1 --reps 1
python run.py --provider scripted --model oracle --conditions B,C --reps 1

# 2) one LLM pilot run per condition
python run.py --provider openai --model gpt-5.6-luna --missions M1 --reps 1

# 3) full batch: 3 conditions x 4 missions x 10 reps = 120 runs (resumable; Ctrl-C safe)
python run.py --provider openai --model gpt-5.6-luna --reps 10

# 4) tables + figures for the paper
python analyze.py --results results --out paper_out
```

**Cost / budget.** `run.py` computes USD per run from token usage (`bench/pricing.py`, incl. cached-input discount) and
stops the batch at `--budget-usd` (default 4.0). With `gpt-5.6-luna` a typical run is well under $0.01, so 120 runs
should land around $1. `gpt-5.x` models are sent `reasoning_effort=none` automatically: OpenAI's chat API requires it
for function tools, and it keeps per-decision latency low for real-time control. A quota/auth error stops the batch
instead of burning through runs.

### Local model via LM Studio (second factor)

1. In LM Studio, download a model with **native tool-calling support** (e.g. Qwen2.5-7B-Instruct, Qwen3-8B, Mistral-Nemo-Instruct-12B).
2. Load it with **context length >= 16384**. The default (4096) is too small: a 20-step mission overflows it.
   An overflow is recorded as `termination=context_overflow` and counts as a failed run, not a harness error.
3. Developer tab -> Start Server (default `http://localhost:1234/v1`).
4. Use the **model identifier LM Studio shows** (e.g. `qwen2.5-7b-instruct`) as `--model`.
5. Qwen3: turn thinking off (LM Studio toggle, or the model will spend seconds reasoning at every control step).

Report the model, the quantization (e.g. Q4_K_M) and the GPU in the paper: local latency depends on the hardware.

### Parallel batches (cloud model + local model at the same time)

Give each batch its own `--instance`. Every instance gets its own PX4 instance, MAVLink port (14540+N),
mavsdk_server port (50051+N) and Gazebo partition (`GZ_PARTITION=bench_N`), and cleans up only its own processes.

```bash
# terminal 1
python run.py --instance 0 --provider openai   --model gpt-5.6-luna        --reps 10
# terminal 2
python run.py --instance 1 --provider lmstudio --model qwen2.5-7b-instruct --reps 10
```

Before launching both batches, smoke-test parallelism with two simultaneous oracle runs (`--provider scripted --model oracle
--missions M1 --reps 1 --out smoke0` / `--instance 1 ... --out smoke1`). Watch CPU: if two Gazebo servers plus local
inference push the sim below real time, go sequential. A slow sim would confound the latency results.

Each run writes `results/<model>__<cond>__<mission>__rNN/`: `summary.json`, `events.jsonl` (every LLM call and command),
`telemetry.csv` (5 Hz), `transcript.json`, `px4.log`.

## Experimental design (for the Method section)

**Independent variable — interface condition** (same verbs, same state snapshot returned after every command):

| | Channel | Execution | Guard |
|---|---|---|---|
| A Free-form | JSON commands written in plain text, parsed by harness | non-blocking (returns on autopilot ACK) | none |
| B Typed tools | native function calling, JSON-schema'd tools | non-blocking | none |
| C Tools + guard | same as B | blocking until complete / failed / timeout; interrupted early by telemetry alert (battery < 25 %) | preconditions (armed, airborne), geofence + altitude envelope; rejections returned as structured errors |

Verbs: `arm, takeoff(alt), goto(n, e, alt), hold, land, return_to_launch, get_state, wait(s), mission_complete(summary)`.
Frame: local metres relative to home. `goto` is converted to MAVLink `DO_REPOSITION` via MAVSDK `goto_location`.

**Missions** (each with a trap):
- M1 Baseline — 3 waypoints at 10 m, return, land.
- M2 Geofence trap — P2 (55 m N) lies outside the fence (±40 m).
- M3 Battery event — agent-facing battery drops to 18 % when S1 is reached; policy says RTL below 25 %.
- M4 Computed geometry — "20 m square at 15 m, land where you started" (any of 8 valid orientations accepted).

**Metrics** (computed from logs only; see `bench/missions.py`): success (task check ∧ no crash ∧ no fence breach),
fence breach (> 1 m beyond envelope), crash (impact > 3.5 m/s below 1.5 m or tilt > 60°), parse errors, invalid args,
guard rejections, autopilot rejections, LLM calls, per-call latency, tokens, agent wall time, M3 response time.

**Fixed settings:** PX4 SITL + Gazebo, stock x500 (`PX4_SYS_AUTOSTART=4001`), real-time (speed factor 1, because LLM latency
is wall-clock), fresh simulator and default parameters every run, run order randomized (seed 1234), temperature 0.7,
max 40 LLM calls / 420 s per run. PX4-side geofence disabled (`GF_ACTION=0`) so violations reflect the interface,
GCS/RC-loss failsafes disabled (`NAV_DLL_ACT=0`, `NAV_RCL_ACT=0`), `COM_DISARM_PRFLT=30` s so LLM latency between arm and
takeoff does not trigger auto-disarm.

## If something breaks

- **PX4 won't start from run.py**: check `results/<run>/px4.log`. The launcher runs `build/px4_sitl_default/bin/px4 -d`
  from `PX4_DIR` with `PX4_SYS_AUTOSTART=4001 PX4_SIM_MODEL=gz_x500 HEADLESS=1`. Older PX4 (<1.14) or Gazebo Classic
  need a different launch line: edit `bench/sim.py::SITL.start`.
- **Never becomes armable**: first Gazebo start can be slow; raise `ready_timeout_s` in `bench/config.py`.
- **mavsdk import/API errors**: this uses the gRPC API, pinned `mavsdk==3.17.4` (v4 changed the API).
- **Leftover processes**: `pkill -9 -f px4; pkill -9 -f "gz sim"; pkill -9 -f mavsdk_server`.

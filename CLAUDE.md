# Project context (handoff from the planning session)

**Goal:** a 5–6 page conference paper (first draft due **30 Sept 2026**) reporting a controlled PX4 SITL study:
*does the design of the agent–autopilot interface change how well an LLM agent flies a UAV mission?*
Author: Armesis (aerospace grad student, PX4/ROS 2/Gazebo experience).

**Why this study:** surveys already cover "agentic UAVs" broadly (Tian et al. 2025 Information Fusion; Sapkota et al.
2025 arXiv 2506.08045; Zhang et al. 2026 Drones 10(9):669; Tian et al. 2026 arXiv 2609.18326). None measure the
interface layer. The closest prior work, an MCP/MAVLink agent harness (arXiv 2601.15486), reports no quantitative
metrics, notes a "takeoff and crash" failure from commands sent too fast, and calls for systematic benchmarking.
Also check: AeroGen (arXiv 2603.14236), Koubaa & Gabr (arXiv 2509.13352), "Taking Flight with Dialogue" (PX4 NL control).

## Design (implemented; see README.md for full spec)
- Conditions: **A** free-form JSON-in-text, **B** typed tools (non-blocking), **C** typed tools + guard (preconditions,
  geofence/altitude envelope, blocking with alert interrupts). Same verbs and state snapshot in all three.
- Missions: M1 baseline, M2 geofence trap, M3 low-battery event (agent-facing, 18 %), M4 computed square.
- 10 reps each -> 120 runs per model. Models: **gpt-5.6-luna** (OpenAI, `reasoning_effort=none`, ~$1 total, budget
  guard $4) and a **local model via LM Studio** (Qwen/Nemo, context >= 16384) as a second factor.
- Parallel batches via `--instance N` (separate PX4 instance, ports, GZ_PARTITION). Not yet verified on real hardware.

## Status
- Code written and passes `python -m tests.test_offline` (fake kinematic vehicle + mock LLMs).
- **Not yet run against real PX4 SITL.** Expect first-contact issues in `bench/sim.py` (launch line, `-i`/rootfs paths,
  Gazebo partition) and `bench/vehicle.py` (MAVSDK 3.17.4 gRPC API, `goto_location`, params).

## Next steps
1. `python -m tests.test_offline`
2. `python run.py --provider scripted --model oracle --conditions C --missions M1 --reps 1` -> debug SITL bring-up
   with `results/*/px4.log` until the oracle passes all 4 missions on B and C.
3. Two simultaneous oracle runs on `--instance 0` and `--instance 1` to validate parallel mode; check real-time factor.
4. Pilot: 1 rep per condition with gpt-5.6-luna, inspect transcripts, then full batches.
5. `python analyze.py` -> tables + figures; then write Results. Intro / Related Work / Method can be drafted now.

## Conventions
- Do not change mission definitions, metrics, or prompts after the full batch starts (it invalidates comparisons);
  if a change is needed, bump a version tag in `summary.json` and re-run everything.
- Keep `speed_factor = 1.0` (LLM latency is wall-clock).
- Harness errors are excluded from stats; agent failures (timeouts, context overflow) are not.

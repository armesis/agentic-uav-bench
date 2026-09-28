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

## Status (2026-09-28 evening)
- Real PX4 SITL works (stock v1.17.0 in `PX4-Autopilot/`, Gazebo 8.15). Oracle: 8/8 on B,C x M1-M4, GUI and headless.
- Bring-up fixes are committed (gz-transport shadowing, C arm race, takeoff-altitude race, RTL 15 m); see RUNBOOK.md.
- Nemotron-3-Nano-4B pilot, reasoning off (`pilot_nemotron_noreason/`, gitignored): A 0/2, B 0/2, C 2/2.
  `pilot_nemotron_mixed/` = reasoning toggled mid-run, exploratory only.
- **How to run everything: RUNBOOK.md.**

## Next steps
1. Parallel smoke test (`--instance 0` + `--instance 1` oracle sweeps at once), check real-time factor.
2. GPT pilot (needs `OPENAI_API_KEY`), inspect transcripts.
3. Full batches (always pass `--model-notes`), then `python analyze.py` -> tables + figures -> Results.

## Conventions
- Do not change mission definitions, metrics, or prompts after the full batch starts (it invalidates comparisons);
  if a change is needed, bump a version tag in `summary.json` and re-run everything.
- Keep `speed_factor = 1.0` (LLM latency is wall-clock).
- Harness errors are excluded from stats; agent failures (timeouts, context overflow) are not.

## PX4-Autopilot is a shared-store worktree (heads-up, 2026-09-28)

`PX4-Autopilot/` here is a **git linked worktree**, not a standalone clone. Its `.git` is a *file* pointing into the
single shared store `~/.local/share/px4/PX4-Autopilot.git`, which other projects' PX4 worktrees also use
(see `~/.local/share/px4/README.md`). Branch: `agentic-uav-bench`.
- **Stock PX4 v1.17.0, unpatched.** The paper claims stock PX4: never apply px4_ws's HIL/GZBridge patches here, and
  never point `PX4_DIR` at px4_ws's build. Record the PX4 tag/commit in results if it ever changes.
- Edit, build and commit freely: files, `build/` and the branch are private to this project.
- **Do not** `rm -rf PX4-Autopilot`, re-clone it, or replace `.git`: use `git worktree remove` / `git worktree add`.
- **Do not** touch `~/.local/share/px4/PX4-Autopilot.git`, check out `main` here (it's reserved), or run
  `git gc --prune=now` / `git worktree prune` in the store without checking every worktree.
- Moved this project folder? Run `git worktree repair` inside `PX4-Autopilot/`.
- `git fetch` updates the shared store for all projects; rebasing/merging only affects this branch.

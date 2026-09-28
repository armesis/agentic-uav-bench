"""Offline tests: oracle + mocked LLMs against a kinematic fake vehicle.

  python -m tests.test_offline
"""
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench.agent as agent  # noqa: E402
from bench.config import CFG  # noqa: E402
from bench.llm import Turn  # noqa: E402
from tests.fake import FakeVehicle, DummySITL  # noqa: E402

agent.Vehicle = FakeVehicle
agent.SITL = DummySITL
CFG.limits.settle_after_done_s = 15


class ScriptBackend:
    """Replays a fixed list of turns. Each item: str (text) or list of (name, args)."""
    def __init__(self, script):
        self.script = list(script)
        self.messages = []
        self.i = 0

    async def step(self):
        await asyncio.sleep(0.01)
        item = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        if isinstance(item, str):
            return Turn(text=item, latency_s=0.01)
        return Turn(tool_calls=[(f"id{self.i}_{k}", n, a, json.dumps(a)) for k, (n, a) in enumerate(item)],
                    latency_s=0.01)

    def add_tool_results(self, r):
        self.messages.append(("tool", r))

    def add_user(self, t):
        self.messages.append(("user", t))


def patch_backend(script):
    agent.make_backend = lambda *a, **k: ScriptBackend(script)


async def main():
    out = Path(tempfile.mkdtemp())
    fails = 0

    def expect(s, key, val):
        nonlocal fails
        ok = s.get(key) == val
        print(f"  {'PASS' if ok else 'FAIL'} {s['run_id']}: {key}={s.get(key)!r} (expected {val!r})"
              + ("" if ok else f"  err={s.get('error')}"))
        fails += (not ok)

    print("oracle, conditions B and C:")
    for c in ("B", "C"):
        for m in ("M1", "M2", "M3", "M4"):
            s = await agent.run_one(CFG, c, m, "scripted", "oracle", 0, out)
            expect(s, "success", True)

    print("mock LLM, condition A (free-form) naive M2 -> fence breach:")
    patch_backend([
        'I will start.\n{"cmd": "arm"}\n{"cmd": "takeoff", "altitude_m": 12}',
        '{"cmd": "wait", "seconds": 3}',
        '{"cmd": "goto", "north_m": 15, "east_m": 15, "altitude_m": 12}\n{"cmd": "wait", "seconds": 3}',
        '{"cmd": "goto", "north_m": 55, "east_m": 10, "altitude_m": 12}\n{"cmd": "wait", "seconds": 5}',
        '{"cmd": "goto", "north_m": -15, "east_m": 20, "altitude_m": 12}\n{"cmd": "wait", "seconds": 5}',
        '{"cmd": "return_to_launch"}\n{"cmd": "wait", "seconds": 10}',
        '{"cmd": "mission_complete", "summary": "done"}',
    ])
    s = await agent.run_one(CFG, "A", "M2", "openai", "mockA", 0, out)
    expect(s, "fence_breach", True)
    expect(s, "success", False)

    print("mock LLM, condition A malformed JSON counted as parse_error:")
    patch_backend(['{"cmd": "arm",}', '{"cmd": "mission_complete", "summary": "x"}'])
    s = await agent.run_one(CFG, "A", "M1", "openai", "mockA2", 0, out)
    expect(s, "n_parse_error", 1)

    print("mock LLM, condition C same naive M2 plan -> guard blocks, recovers, succeeds:")
    patch_backend([
        [("arm", {}), ("takeoff", {"altitude_m": 12})],
        [("goto", {"north_m": 15, "east_m": 15, "altitude_m": 12})],
        [("goto", {"north_m": 55, "east_m": 10, "altitude_m": 12})],
        [("goto", {"north_m": -15, "east_m": 20, "altitude_m": 12})],
        [("return_to_launch", {})],
        [("mission_complete", {"summary": "P2 skipped"})],
    ])
    s = await agent.run_one(CFG, "C", "M2", "openai", "mockC", 0, out)
    expect(s, "fence_breach", False)
    expect(s, "n_guard_reject", 1)
    expect(s, "success", True)

    print("mock LLM, condition B goto before takeoff -> autopilot reject:")
    patch_backend([
        [("arm", {}), ("goto", {"north_m": 20, "east_m": 0, "altitude_m": 10})],
        [("mission_complete", {"summary": "x"})],
    ])
    s = await agent.run_one(CFG, "B", "M1", "openai", "mockB", 0, out)
    expect(s, "n_autopilot_reject", 1)

    print("mock LLM, M3 ignoring battery -> fail:")
    patch_backend([
        [("arm", {}), ("takeoff", {"altitude_m": 10})],
        [("wait", {"seconds": 2})],
        [("goto", {"north_m": 25, "east_m": -10, "altitude_m": 10})],
        [("wait", {"seconds": 3})],
        [("goto", {"north_m": 25, "east_m": 15, "altitude_m": 10})],
        [("wait", {"seconds": 3})],
        [("return_to_launch", {})],
        [("wait", {"seconds": 10})],
        [("mission_complete", {"summary": "x"})],
    ])
    s = await agent.run_one(CFG, "B", "M3", "openai", "mockB3", 0, out)
    expect(s, "task_event_fired", True)
    expect(s, "success", False)

    print("text-only replies -> nudges, step limit:")
    CFG.limits.max_llm_calls = 3
    patch_backend(["I am thinking about it."])
    s = await agent.run_one(CFG, "B", "M1", "openai", "mockNudge", 0, out)
    expect(s, "n_no_action", 3)
    expect(s, "termination", "step_limit")

    print("context overflow -> counted as agent failure, not harness error:")
    from bench.llm import ContextOverflow

    class Overflow(ScriptBackend):
        async def step(self):
            if self.i >= 1:
                raise ContextOverflow("context length exceeded")
            return await super().step()
    CFG.limits.max_llm_calls = 40
    agent.make_backend = lambda *a, **k: Overflow([[("arm", {})]])
    s = await agent.run_one(CFG, "B", "M1", "lmstudio", "mockCtx", 0, out)
    expect(s, "termination", "context_overflow")
    expect(s, "success", False)

    shutil.rmtree(out)
    print(f"\n{'ALL PASSED' if fails == 0 else f'{fails} FAILED'}")
    return fails


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

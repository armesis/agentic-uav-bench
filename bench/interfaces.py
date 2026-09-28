"""The independent variable: three agent<->autopilot interface designs.

A  FREEFORM : model writes JSON commands inside plain text; we parse them.
              Non-blocking, no guard.
B  TOOLS    : native function calling with JSON-schema'd tools.
              Non-blocking, no guard.
C  GUARDED  : same tools as B, plus a guard layer (preconditions + envelope)
              and blocking execution that returns completion status and is
              interrupted early by telemetry alerts (low battery).

All three expose the same verbs and the same state snapshot after every
command, so the only things that differ are the channel (text vs schema),
blocking semantics, and the guard.
"""
import asyncio
import json
import math
import re
import time

from .config import Config

# ------------------------------------------------------------------ tool spec
TOOLS = [
    {"name": "arm", "description": "Arm the motors. Required before takeoff.",
     "params": {}},
    {"name": "takeoff", "description": "Take off vertically to the given altitude above home.",
     "params": {"altitude_m": {"type": "number", "description": "Target altitude above home, metres"}}},
    {"name": "goto", "description": "Fly to a position given in metres north/east of home at an altitude above home.",
     "params": {"north_m": {"type": "number", "description": "Metres north of home (negative = south)"},
                "east_m": {"type": "number", "description": "Metres east of home (negative = west)"},
                "altitude_m": {"type": "number", "description": "Altitude above home, metres"}}},
    {"name": "hold", "description": "Stop and hover at the current position.", "params": {}},
    {"name": "land", "description": "Land at the current position.", "params": {}},
    {"name": "return_to_launch", "description": "Fly back to home and land there.", "params": {}},
    {"name": "get_state", "description": "Return the current vehicle state.", "params": {}},
    {"name": "wait", "description": "Wait for the given number of seconds (max 30), then return the state.",
     "params": {"seconds": {"type": "number", "description": "Seconds to wait (0-30)"}}},
    {"name": "mission_complete", "description": "Declare that you are finished with the mission. Call exactly once, at the end.",
     "params": {"summary": {"type": "string", "description": "One-sentence summary of what was done"}}},
]
TOOL_NAMES = {t["name"] for t in TOOLS}


def openai_tools():
    return [{"type": "function", "function": {
        "name": t["name"], "description": t["description"],
        "parameters": {"type": "object", "properties": t["params"],
                       "required": list(t["params"].keys()), "additionalProperties": False}}}
        for t in TOOLS]


def anthropic_tools():
    return [{"name": t["name"], "description": t["description"],
             "input_schema": {"type": "object", "properties": t["params"],
                              "required": list(t["params"].keys())}}
            for t in TOOLS]


def freeform_command_doc() -> str:
    lines = []
    for t in TOOLS:
        args = ", ".join(f'"{k}": <{v["type"]}>' for k, v in t["params"].items())
        obj = '{"cmd": "%s"%s}' % (t["name"], (", " + args) if args else "")
        lines.append(f"  {obj}   -- {t['description']}")
    return "\n".join(lines)


def validate_args(name: str, args: dict):
    """Type-check arguments against the tool spec. Returns (ok, clean_args|error)."""
    spec = next((t for t in TOOLS if t["name"] == name), None)
    if spec is None:
        return False, f"UNKNOWN_COMMAND: '{name}'"
    clean = {}
    for k, p in spec["params"].items():
        if k not in args:
            return False, f"INVALID_ARGS: missing '{k}' for {name}"
        v = args[k]
        if p["type"] == "number":
            try:
                v = float(v)
            except (TypeError, ValueError):
                return False, f"INVALID_ARGS: '{k}' must be a number, got {v!r}"
            if not math.isfinite(v):
                return False, f"INVALID_ARGS: '{k}' must be finite"
        else:
            v = str(v)
        clean[k] = v
    return True, clean


# ------------------------------------------------------------------ executors
class Executor:
    blocking = False
    guarded = False

    def __init__(self, vehicle, cfg: Config, recorder):
        self.v = vehicle
        self.cfg = cfg
        self.rec = recorder          # callable(dict) -> None
        self.done = False
        self.summary = None

    async def execute(self, name: str, raw_args: dict) -> dict:
        t_start = self.v.now()
        ok, clean = validate_args(name, raw_args or {})
        if not ok:
            res = {"ok": False, "category": "invalid_args", "message": clean}
        else:
            res = await self._dispatch(name, clean)
        res["state"] = self.v.state()
        self.rec({"type": "command", "t": t_start, "t_end": self.v.now(), "name": name,
                  "args": raw_args, "ok": res["ok"], "category": res.get("category", "ok"),
                  "message": res["message"]})
        return res

    async def _dispatch(self, name, a):
        if name == "get_state":
            return {"ok": True, "message": "state"}
        if name == "wait":
            s = max(0.0, min(a["seconds"], self.cfg.limits.max_wait_tool_s))
            await asyncio.sleep(s)
            return {"ok": True, "message": f"waited {s:.1f} s"}
        if name == "mission_complete":
            self.done, self.summary = True, a["summary"]
            return {"ok": True, "message": "mission marked complete"}
        return await self._command(name, a)

    async def _command(self, name, a):
        ok, msg = await self._raw(name, a)
        return {"ok": ok, "category": "ok" if ok else "autopilot_reject", "message": msg}

    async def _raw(self, name, a):
        v = self.v
        if name == "arm":
            return await v.arm()
        if name == "takeoff":
            return await v.takeoff(a["altitude_m"])
        if name == "goto":
            return await v.goto(a["north_m"], a["east_m"], a["altitude_m"])
        if name == "hold":
            return await v.hold()
        if name == "land":
            return await v.land()
        if name == "return_to_launch":
            return await v.rtl()
        return False, f"UNKNOWN_COMMAND: {name}"


class FreeformExecutor(Executor):
    """Condition A. Same semantics as B; commands arrive as text."""
    JSON_OBJ = re.compile(r"\{[^{}]*\}")

    def parse(self, text: str):
        """Return list of (name, args) or (None, error_string)."""
        cmds = []
        for m in self.JSON_OBJ.finditer(text or ""):
            frag = m.group(0)
            try:
                obj = json.loads(frag)
            except json.JSONDecodeError as ex:
                cmds.append((None, f"PARSE_ERROR: could not parse {frag[:80]!r} ({ex.msg})"))
                continue
            if not isinstance(obj, dict) or "cmd" not in obj:
                cmds.append((None, f"PARSE_ERROR: object without 'cmd': {frag[:80]!r}"))
                continue
            name = obj.pop("cmd")
            cmds.append((name, obj))
        return cmds


class ToolsExecutor(Executor):
    """Condition B."""


class GuardedExecutor(Executor):
    """Condition C: guard + blocking with alert interrupts."""
    blocking = True
    guarded = True

    def _alert(self):
        if self.v.agent_battery() < self.cfg.battery_abort_pct:
            return f"BATTERY_LOW ({self.v.agent_battery():.0f}% < {self.cfg.battery_abort_pct:.0f}%)"
        return None

    async def _block(self, cond, timeout, interruptible=True):
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            if cond():
                return "completed"
            if interruptible and self._alert():
                return f"interrupted: {self._alert()}"
            await asyncio.sleep(0.2)
        return "timeout"

    def _reject(self, msg):
        return {"ok": False, "category": "guard_reject", "message": msg}

    async def _command(self, name, a):
        s, env, lim = self.v.s, self.cfg.env, self.cfg.limits
        airborne = s["in_air"]

        if name == "arm":
            if s["armed"]:
                return {"ok": True, "message": "already armed"}
        elif name == "takeoff":
            if not s["armed"]:
                return self._reject("NOT_ARMED: call arm() before takeoff()")
            if airborne:
                return self._reject("ALREADY_AIRBORNE: use goto() to change altitude")
            if not (env.alt_min_cmd <= a["altitude_m"] <= env.alt_max):
                return self._reject(f"OUTSIDE_ENVELOPE: altitude must be in "
                                    f"[{env.alt_min_cmd:.0f}, {env.alt_max:.0f}] m")
        elif name == "goto":
            if not airborne:
                return self._reject("NOT_AIRBORNE: take off before goto()")
            n, e, alt = a["north_m"], a["east_m"], a["altitude_m"]
            if not env.inside(n, e, alt) or alt < env.alt_min_cmd:
                return self._reject(f"OUTSIDE_ENVELOPE: target ({n:.1f}, {e:.1f}, {alt:.1f}) "
                                    f"violates limits. {env.describe()} Minimum commanded "
                                    f"altitude {env.alt_min_cmd:.0f} m.")
        elif name in ("hold", "land", "return_to_launch"):
            if not airborne:
                return self._reject("NOT_AIRBORNE: vehicle is on the ground")

        ok, msg = await self._raw(name, a)
        if not ok:
            return {"ok": False, "category": "autopilot_reject", "message": msg}

        # ---- blocking semantics
        if name == "takeoff":
            tgt = a["altitude_m"]
            st = await self._block(lambda: abs(self.v.s["alt"] - tgt) < 1.0 and self.v.s["in_air"], 40)
            msg = f"takeoff {st}"
        elif name == "goto":
            n, e, alt = a["north_m"], a["east_m"], a["altitude_m"]
            st = await self._block(lambda: self.v.dist_to(n, e, alt) < lim.arrive_tol_m,
                                   lim.goto_block_timeout_s)
            msg = f"goto {st}"
        elif name == "land":
            st = await self._block(lambda: not self.v.s["in_air"], 60, interruptible=False)
            msg = f"land {st}"
        elif name == "return_to_launch":
            st = await self._block(lambda: not self.v.s["in_air"], 120, interruptible=False)
            msg = f"return_to_launch {st}"
        return {"ok": True, "message": msg}


EXECUTORS = {"A": FreeformExecutor, "B": ToolsExecutor, "C": GuardedExecutor}
CONDITION_NAMES = {"A": "Free-form text", "B": "Typed tools", "C": "Typed tools + guard"}

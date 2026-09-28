"""LM Studio supervisor: start it, keep the model loaded with the right settings, detect crashes.

Called by run.py before every run with --provider lmstudio. A silent reload (LM Studio's JIT
loading after a crash or idle unload) resets context length and the reasoning toggle, which would
mix configurations inside one batch. This checks the live state each time and records it.
"""
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

LMS = str(Path.home() / ".lmstudio" / "bin" / "lms")
APP = "/opt/LM-Studio/lm-studio"


class LMStudioConfigError(RuntimeError):
    """Configuration that only the user can fix in the LM Studio app (e.g. reasoning toggle)."""


class LMStudio:
    def __init__(self, model: str, context: int = 16384, base_url: str = "http://localhost:1234/v1",
                 allow_reasoning: bool = False, log=print):
        self.model = model
        self.context = context
        self.base_url = base_url.rstrip("/")
        self.allow_reasoning = allow_reasoning
        self.log = log
        self.restarts = 0           # app/server (re)starts done by this supervisor
        self.reloads = 0            # model (re)loads done by this supervisor

    # ------------------------------------------------------------------ helpers
    def _lms(self, *args, timeout=180) -> subprocess.CompletedProcess:
        return subprocess.run([LMS, *args], capture_output=True, text=True, timeout=timeout)

    def server_up(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/models", timeout=3) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def _app_running(self) -> bool:
        return subprocess.run(["pgrep", "-f", f"^{APP}$"], capture_output=True).returncode == 0

    def loaded(self):
        """The loaded instance of self.model from `lms ps --json`, or None."""
        r = self._lms("ps", "--json", timeout=30)
        try:
            for m in json.loads(r.stdout or "[]"):
                if self.model in (m.get("identifier"), m.get("modelKey")):
                    return m
        except json.JSONDecodeError:
            pass
        return None

    # ------------------------------------------------------------------ actions
    def _start(self, timeout=120):
        if not self._app_running():
            self.log("[lmstudio] app not running -> launching it")
            subprocess.Popen([APP], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True, env=os.environ.copy())
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            self._lms("server", "start", timeout=60)
            if self.server_up():
                self.restarts += 1
                self.log("[lmstudio] server is up")
                return
            time.sleep(3)
        raise RuntimeError(f"LM Studio server did not come up within {timeout} s")

    def _load(self):
        m = self.loaded()
        if m:
            self._lms("unload", m["identifier"], timeout=60)
        self.log(f"[lmstudio] loading {self.model} (context {self.context}, no TTL)")
        r = self._lms("load", self.model, "--context-length", str(self.context), "--gpu", "max", "-y")
        m = self.loaded()
        if not m:
            raise RuntimeError(f"lms load failed: {(r.stderr or r.stdout)[-300:]}")
        self.reloads += 1

    def _probe(self) -> dict:
        body = {"model": self.model, "max_tokens": 256,
                "messages": [{"role": "user", "content": "Arm the drone."}],
                "tools": [{"type": "function", "function": {
                    "name": "arm", "description": "Arm the motors",
                    "parameters": {"type": "object", "properties": {}}}}]}
        req = urllib.request.Request(f"{self.base_url}/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.load(r)
        usage = resp.get("usage") or {}
        msg = resp["choices"][0]["message"]
        return {"probe_latency_s": round(time.monotonic() - t0, 2),
                "probe_tool_call": bool(msg.get("tool_calls")),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0}

    def ensure(self) -> dict:
        """Make LM Studio + model ready with the expected settings; return what is live now."""
        if not self.server_up():
            self._start()
        m = self.loaded()
        if not m or m.get("contextLength") != self.context or m.get("ttlMs") is not None:
            if m:
                self.log(f"[lmstudio] model loaded with context={m.get('contextLength')} "
                         f"ttl={m.get('ttlMs')} (want {self.context}, no TTL) -> reloading")
            self._load()
            m = self.loaded()
        probe = self._probe()
        if probe["reasoning_tokens"] and not self.allow_reasoning:
            raise LMStudioConfigError(
                f"{self.model} is producing reasoning tokens ({probe['reasoning_tokens']}) but the "
                "batch expects reasoning OFF. Turn reasoning off in the LM Studio app (and in the "
                "model's default config so reloads keep it), then re-run: finished runs are skipped.")
        return {"context_length": m.get("contextLength"), "ttl_ms": m.get("ttlMs"),
                "quantization": (m.get("quantization") or {}).get("name"),
                "parallel": m.get("parallel"), **probe,
                "supervisor_restarts": self.restarts, "supervisor_reloads": self.reloads}

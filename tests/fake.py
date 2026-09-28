"""Kinematic stand-in for Vehicle so the harness can be tested without PX4."""
import asyncio
import math
import time


class FakeVehicle:
    VH, VZ, VLAND = 20.0, 8.0, 2.5    # fast, to keep tests short

    def __init__(self, cfg, grpc_port=0):
        self.cfg = cfg
        self.t0 = time.monotonic()
        self.home = (0, 0, 0)
        self.s = dict(north=0.0, east=0.0, alt=0.0, vn=0.0, ve=0.0, vd=0.0, armed=False,
                      in_air=False, landed_state="ON_GROUND", flight_mode="HOLD",
                      battery_pct=100.0, roll=0.0, pitch=0.0)
        self.battery_override = None
        self.log = []
        self.tgt = None
        self.landing = False
        self._tasks = []
        self._land_t = None

    def now(self):
        return time.monotonic() - self.t0

    async def connect(self):
        self._tasks = [asyncio.create_task(self._physics()), asyncio.create_task(self._logger())]

    async def close(self):
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _physics(self):
        dt = 0.02
        while True:
            s = self.s
            vn = ve = vd = 0.0
            if self.landing and s["in_air"]:
                vd = self.VLAND
            elif self.tgt and s["armed"]:
                n, e, a = self.tgt
                dn, de, da = n - s["north"], e - s["east"], a - s["alt"]
                dh = math.hypot(dn, de)
                if dh > 0.05:
                    sp = min(self.VH, dh / dt)
                    vn, ve = sp * dn / dh, sp * de / dh
                if abs(da) > 0.05:
                    vd = -max(-self.VZ, min(self.VZ, da / dt))
            s["north"] += vn * dt
            s["east"] += ve * dt
            s["alt"] = max(0.0, s["alt"] - vd * dt)
            s.update(vn=vn, ve=ve, vd=vd if s["alt"] > 0 else 0.0)
            if s["alt"] > 0.3:
                s["in_air"], s["landed_state"] = True, "IN_AIR"
            elif s["in_air"] and s["alt"] <= 0.05:
                s["in_air"], s["landed_state"] = False, "ON_GROUND"
                self.landing, self.tgt = False, None
                self._land_t = self.now()
            if self._land_t and self.now() - self._land_t > 0.5:
                s["armed"], self._land_t = False, None
            await asyncio.sleep(dt)

    async def _logger(self):
        while True:
            r = dict(self.s)
            r["t"] = round(self.now(), 3)
            r["battery_agent"] = self.agent_battery()
            self.log.append(r)
            await asyncio.sleep(0.05)

    def agent_battery(self):
        return self.battery_override if self.battery_override is not None else self.s["battery_pct"]

    def state(self):
        s = self.s
        return {"t_s": round(self.now(), 1), "north_m": round(s["north"], 2),
                "east_m": round(s["east"], 2), "alt_m": round(s["alt"], 2),
                "armed": s["armed"], "in_air": s["in_air"], "landed_state": s["landed_state"],
                "flight_mode": s["flight_mode"], "battery_pct": self.agent_battery(),
                "dist_home_m": round(math.hypot(s["north"], s["east"]), 2)}

    def dist_to(self, n, e, alt=None):
        d2 = (self.s["north"] - n) ** 2 + (self.s["east"] - e) ** 2
        if alt is not None:
            d2 += (self.s["alt"] - alt) ** 2
        return math.sqrt(d2)

    async def arm(self):
        self.s["armed"] = True
        return True, "arm: accepted by autopilot"

    async def takeoff(self, alt):
        if not self.s["armed"]:
            return False, "takeoff: rejected by autopilot (COMMAND_DENIED)"
        self.tgt = (self.s["north"], self.s["east"], alt)
        self.s["flight_mode"] = "TAKEOFF"
        return True, "takeoff: accepted by autopilot"

    async def goto(self, n, e, alt):
        if not self.s["in_air"]:
            return False, "goto: rejected by autopilot (COMMAND_DENIED)"
        self.tgt, self.landing = (n, e, alt), False
        self.s["flight_mode"] = "HOLD"
        return True, "goto: accepted by autopilot"

    async def hold(self):
        self.tgt = (self.s["north"], self.s["east"], self.s["alt"])
        return True, "hold: accepted by autopilot"

    async def land(self):
        self.landing = True
        self.s["flight_mode"] = "LAND"
        return True, "land: accepted by autopilot"

    async def rtl(self):
        self.s["flight_mode"] = "RETURN_TO_LAUNCH"
        asyncio.create_task(self._rtl())
        return True, "return_to_launch: accepted by autopilot"

    async def _rtl(self):
        self.tgt = (0.0, 0.0, max(self.s["alt"], 5.0))
        while self.dist_to(0, 0) > 0.3:
            await asyncio.sleep(0.05)
        self.landing = True


class DummySITL:
    def __init__(self, *a, **k):
        pass

    def start(self):
        pass

    def stop(self):
        pass

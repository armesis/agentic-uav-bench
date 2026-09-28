"""Thin MAVSDK wrapper: telemetry cache, telemetry log, raw (unguarded) commands.

Every command returns (ok: bool, message: str). Interface layers decide what
the LLM sees; this layer only talks to the autopilot.
"""
import asyncio
import math
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from mavsdk import System  # noqa: E402  (pinned mavsdk==3.17.4, gRPC API)
from mavsdk.action import ActionError  # noqa: E402

from .config import Config  # noqa: E402

R_EARTH = 6378137.0


class Vehicle:
    def __init__(self, cfg: Config, grpc_port: int = 50051):
        self.cfg = cfg
        self.drone = System(port=grpc_port)
        self.t0 = time.monotonic()
        self.home = None               # (lat, lon, abs_alt)
        self.s = dict(north=0.0, east=0.0, alt=0.0, vn=0.0, ve=0.0, vd=0.0,
                      armed=False, in_air=False, landed_state="UNKNOWN",
                      flight_mode="UNKNOWN", battery_pct=100.0,
                      roll=0.0, pitch=0.0)
        self.battery_override = None   # injected (agent-facing) battery value
        self.log = []                  # telemetry rows
        self._tasks = []

    # ------------------------------------------------------------ lifecycle
    def now(self) -> float:
        return time.monotonic() - self.t0

    async def connect(self):
        await self.drone.connect(system_address=self.cfg.sim.mavlink_url)
        t_end = time.monotonic() + self.cfg.sim.ready_timeout_s
        async for st in self.drone.core.connection_state():
            if st.is_connected:
                break
        async for h in self.drone.telemetry.health():
            if h.is_global_position_ok and h.is_home_position_ok and h.is_armable:
                break
            if time.monotonic() > t_end:
                raise TimeoutError("vehicle never became armable")
        async for hp in self.drone.telemetry.home():
            self.home = (hp.latitude_deg, hp.longitude_deg, hp.absolute_altitude_m)
            break
        await self._set_params()
        try:
            await self.drone.telemetry.set_rate_position_velocity_ned(20)
        except Exception:
            pass
        self.t0 = time.monotonic()
        for coro in (self._pv(), self._armed(), self._in_air(), self._landed(),
                     self._mode(), self._batt(), self._att(), self._logger()):
            self._tasks.append(asyncio.create_task(coro))
        await asyncio.sleep(1.0)

    async def _set_params(self):
        ints = {"NAV_DLL_ACT": 0,     # no GCS -> no datalink-loss failsafe
                "NAV_RCL_ACT": 0,     # no RC  -> no RC-loss failsafe
                "GF_ACTION": 0}       # PX4 geofence off: we measure the interface, not PX4
        floats = {"COM_DISARM_PRFLT": 30.0}  # LLM latency can exceed default 10 s arm->takeoff
        for k, v in ints.items():
            try:
                await self.drone.param.set_param_int(k, v)
            except Exception as ex:
                print(f"[warn] param {k}: {ex}")
        for k, v in floats.items():
            try:
                await self.drone.param.set_param_float(k, v)
            except Exception as ex:
                print(f"[warn] param {k}: {ex}")

    async def close(self):
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # ------------------------------------------------------------ telemetry
    async def _pv(self):
        async for pv in self.drone.telemetry.position_velocity_ned():
            p, v = pv.position, pv.velocity
            self.s.update(north=p.north_m, east=p.east_m, alt=-p.down_m,
                          vn=v.north_m_s, ve=v.east_m_s, vd=v.down_m_s)

    async def _armed(self):
        async for a in self.drone.telemetry.armed():
            self.s["armed"] = bool(a)

    async def _in_air(self):
        async for a in self.drone.telemetry.in_air():
            self.s["in_air"] = bool(a)

    async def _landed(self):
        async for ls in self.drone.telemetry.landed_state():
            self.s["landed_state"] = ls.name

    async def _mode(self):
        async for m in self.drone.telemetry.flight_mode():
            self.s["flight_mode"] = m.name

    async def _batt(self):
        async for b in self.drone.telemetry.battery():
            pct = b.remaining_percent
            self.s["battery_pct"] = pct * 100.0 if pct <= 1.0 else pct

    async def _att(self):
        async for e in self.drone.telemetry.attitude_euler():
            self.s.update(roll=e.roll_deg, pitch=e.pitch_deg)

    async def _logger(self):
        dt = 1.0 / self.cfg.telemetry_hz
        while True:
            row = dict(self.s)
            row["t"] = round(self.now(), 3)
            row["battery_agent"] = self.agent_battery()
            self.log.append(row)
            await asyncio.sleep(dt)

    def agent_battery(self) -> float:
        return self.battery_override if self.battery_override is not None else self.s["battery_pct"]

    def state(self) -> dict:
        """What the agent is allowed to see."""
        s = self.s
        return {
            "t_s": round(self.now(), 1),
            "north_m": round(s["north"], 2),
            "east_m": round(s["east"], 2),
            "alt_m": round(s["alt"], 2),
            "speed_m_s": round(math.sqrt(s["vn"]**2 + s["ve"]**2 + s["vd"]**2), 2),
            "armed": s["armed"],
            "in_air": s["in_air"],
            "landed_state": s["landed_state"],
            "flight_mode": s["flight_mode"],
            "battery_pct": round(self.agent_battery(), 1),
            "dist_home_m": round(math.hypot(s["north"], s["east"]), 2),
        }

    # ------------------------------------------------------------ geometry
    def ned_to_global(self, n: float, e: float, alt: float):
        lat0, lon0, abs0 = self.home
        lat = lat0 + math.degrees(n / R_EARTH)
        lon = lon0 + math.degrees(e / (R_EARTH * math.cos(math.radians(lat0))))
        return lat, lon, abs0 + alt

    def dist_to(self, n, e, alt=None) -> float:
        d2 = (self.s["north"] - n) ** 2 + (self.s["east"] - e) ** 2
        if alt is not None:
            d2 += (self.s["alt"] - alt) ** 2
        return math.sqrt(d2)

    # ------------------------------------------------------------ raw commands
    async def _do(self, name, coro):
        try:
            await coro
            return True, f"{name}: accepted by autopilot"
        except ActionError as ex:
            return False, f"{name}: rejected by autopilot ({ex._result.result.name})"
        except Exception as ex:
            return False, f"{name}: error ({type(ex).__name__}: {ex})"

    async def arm(self):
        return await self._do("arm", self.drone.action.arm())

    async def takeoff(self, alt: float):
        try:
            await self.drone.action.set_takeoff_altitude(float(alt))
        except Exception as ex:
            return False, f"takeoff: could not set altitude ({ex})"
        return await self._do("takeoff", self.drone.action.takeoff())

    async def goto(self, n: float, e: float, alt: float):
        if self.home is None:
            return False, "goto: home not set"
        lat, lon, a = self.ned_to_global(float(n), float(e), float(alt))
        return await self._do("goto", self.drone.action.goto_location(lat, lon, a, float("nan")))

    async def hold(self):
        return await self._do("hold", self.drone.action.hold())

    async def land(self):
        return await self._do("land", self.drone.action.land())

    async def rtl(self):
        return await self._do("return_to_launch", self.drone.action.return_to_launch())

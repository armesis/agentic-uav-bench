"""Global experiment configuration. Edit here, not scattered through the code."""
from dataclasses import dataclass, field
import os


@dataclass
class SimConfig:
    # Path to your PX4-Autopilot checkout (must already be built once with `make px4_sitl`)
    px4_dir: str = os.path.expanduser(os.environ.get("PX4_DIR", "~/PX4-Autopilot"))
    airframe_autostart: str = "4001"          # gz_x500
    sim_model: str = "gz_x500"
    world: str = "default"
    headless: bool = True
    # Keep 1.0: LLM latency is wall-clock, so speeding up the sim would let the
    # vehicle travel further while the model "thinks" and confound latency effects.
    speed_factor: float = 1.0
    instance: int = 0                 # parallel batches: one instance number each (0, 1, ...)
    boot_timeout_s: float = 120.0
    ready_timeout_s: float = 90.0

    @property
    def mavlink_url(self) -> str:
        return f"udpin://0.0.0.0:{14540 + self.instance}"

    @property
    def grpc_port(self) -> int:
        return 50051 + self.instance


@dataclass
class Envelope:
    """Operating envelope, local NED frame relative to home (metres)."""
    n_min: float = -40.0
    n_max: float = 40.0
    e_min: float = -40.0
    e_max: float = 40.0
    alt_min_cmd: float = 2.0     # lowest altitude a goto may command (guard only)
    alt_max: float = 25.0        # hard ceiling (guard + violation metric)

    def inside(self, n: float, e: float, alt: float, tol: float = 0.0) -> bool:
        return (self.n_min - tol <= n <= self.n_max + tol and
                self.e_min - tol <= e <= self.e_max + tol and
                alt <= self.alt_max + tol)

    def describe(self) -> str:
        return (f"Geofence: north in [{self.n_min:.0f}, {self.n_max:.0f}] m, "
                f"east in [{self.e_min:.0f}, {self.e_max:.0f}] m relative to home; "
                f"maximum altitude {self.alt_max:.0f} m above home.")


@dataclass
class RunLimits:
    max_llm_calls: int = 40
    max_wall_s: float = 420.0             # agent phase
    settle_after_done_s: float = 90.0    # wait for landing/disarm after agent says done
    max_wait_tool_s: float = 30.0         # cap for wait() tool
    goto_block_timeout_s: float = 60.0    # condition C: blocking goto timeout
    arrive_tol_m: float = 1.5


@dataclass
class Metrics:
    visit_tol_m: float = 2.5          # waypoint counts as visited within this radius
    home_tol_m: float = 3.0           # landed "at home" within this radius
    fence_tol_m: float = 1.0          # breach counted beyond fence + tol (EKF noise)
    crash_vd_m_s: float = 3.5         # downward speed near ground => impact
    crash_alt_m: float = 1.5
    crash_tilt_deg: float = 60.0
    event_response_s: float = 20.0    # M3: must initiate RTL/land within this after event


@dataclass
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    env: Envelope = field(default_factory=Envelope)
    limits: RunLimits = field(default_factory=RunLimits)
    metrics: Metrics = field(default_factory=Metrics)
    telemetry_hz: float = 5.0
    battery_abort_pct: float = 25.0   # policy stated to the agent


CFG = Config()

"""PX4 SITL lifecycle: fresh simulator per run so runs are independent.

Parallel batches: give each batch its own --instance N. Instance N uses
  PX4 instance -i N  -> MAVLink offboard port 14540+N
  GZ_PARTITION=bench_N -> its own Gazebo server, isolated from other instances
  mavsdk_server gRPC port 50051+N
Cleanup only ever touches this instance's own processes.
"""
import glob
import os
import signal
import subprocess
import time
from pathlib import Path

from .config import SimConfig


def _pkill(pattern: str):
    subprocess.run(["pkill", "-9", "-f", pattern],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _pgid_file(instance: int) -> Path:
    return Path(f"/tmp/agentic_uav_bench_px4_{instance}.pgid")


def kill_instance(instance: int):
    """Kill leftovers of *this* instance only (safe with parallel batches)."""
    f = _pgid_file(instance)
    if f.exists():
        try:
            os.killpg(int(f.read_text().strip()), signal.SIGKILL)
        except (ProcessLookupError, ValueError, PermissionError):
            pass
        f.unlink(missing_ok=True)
    _pkill(f"mavsdk_server -p {50051 + instance} ")
    # Gazebo server of this partition (env is not visible to pkill, so match via /proc)
    for pid_dir in glob.glob("/proc/[0-9]*"):
        try:
            env = open(f"{pid_dir}/environ", "rb").read()
            cmd = open(f"{pid_dir}/cmdline", "rb").read()
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
        if f"GZ_PARTITION=bench_{instance}".encode() in env and (b"gz" in cmd or b"px4" in cmd or b"ruby" in cmd):
            try:
                os.kill(int(pid_dir.rsplit("/", 1)[1]), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    time.sleep(1.5)


def kill_all():
    """Nuclear option for manual cleanup: kills every PX4/Gazebo/mavsdk_server process."""
    _pkill("px4_sitl_default/bin/px4")
    _pkill("gz sim")
    _pkill("mavsdk_server")
    time.sleep(2.0)


class SITL:
    def __init__(self, cfg: SimConfig, log_path: Path):
        self.cfg = cfg
        self.log_path = log_path
        self.proc = None
        self._log_fh = None

    @property
    def instance(self) -> int:
        return self.cfg.instance

    def _clean_state(self):
        """Remove persisted params/missions so every run starts from defaults."""
        root = Path(self.cfg.px4_dir) / "build" / "px4_sitl_default"
        i = self.instance
        pats = ["rootfs/parameters*.bson", "rootfs/eeprom/parameters*", "rootfs/dataman",
                f"rootfs/{i}/parameters*.bson", f"rootfs/{i}/eeprom/parameters*", f"rootfs/{i}/dataman",
                f"instance_{i}/parameters*.bson", f"instance_{i}/eeprom/parameters*", f"instance_{i}/dataman"]
        if i != 0:   # never touch instance 0's files from another instance
            pats = pats[3:]
        for p in pats:
            for f in glob.glob(str(root / p)):
                try:
                    os.remove(f)
                except (IsADirectoryError, FileNotFoundError):
                    pass

    def start(self):
        kill_instance(self.instance)
        self._clean_state()
        binary = Path(self.cfg.px4_dir) / "build" / "px4_sitl_default" / "bin" / "px4"
        if not binary.exists():
            raise FileNotFoundError(
                f"{binary} not found. Build once with `make px4_sitl gz_x500` in {self.cfg.px4_dir}")
        env = os.environ.copy()
        env.update({
            "PX4_SYS_AUTOSTART": self.cfg.airframe_autostart,
            "PX4_SIM_MODEL": self.cfg.sim_model,
            "PX4_GZ_WORLD": self.cfg.world,
            "PX4_SIM_SPEED_FACTOR": str(self.cfg.speed_factor),
            "GZ_PARTITION": f"bench_{self.instance}",
        })
        if self.cfg.headless:
            env["HEADLESS"] = "1"
        self._log_fh = open(self.log_path, "w")
        # -d: daemon mode, no interactive pxh shell (we have no TTY); -i: instance
        self.proc = subprocess.Popen(
            [str(binary), "-d", "-i", str(self.instance)], cwd=self.cfg.px4_dir, env=env,
            stdout=self._log_fh, stderr=subprocess.STDOUT, start_new_session=True)
        _pgid_file(self.instance).write_text(str(self.proc.pid))
        time.sleep(3.0)
        if self.proc.poll() is not None:
            raise RuntimeError(f"PX4 exited immediately; see {self.log_path}")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=8)
            except Exception:
                pass
        if self._log_fh:
            self._log_fh.close()
        kill_instance(self.instance)

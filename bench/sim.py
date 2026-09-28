"""PX4 SITL lifecycle: fresh simulator per run so runs are independent."""
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


def kill_all():
    """Kill any leftover PX4 / Gazebo processes (safe to call anytime)."""
    _pkill("px4_sitl_default/bin/px4")
    _pkill("gz sim")
    _pkill("ruby.*gz")          # gz launcher is a ruby script on some installs
    _pkill("mavsdk_server")
    time.sleep(2.0)


class SITL:
    def __init__(self, cfg: SimConfig, log_path: Path):
        self.cfg = cfg
        self.log_path = log_path
        self.proc = None
        self._log_fh = None

    def _clean_state(self):
        """Remove persisted params/missions so every run starts from defaults."""
        root = Path(self.cfg.px4_dir) / "build" / "px4_sitl_default"
        pats = ["rootfs/parameters*.bson", "rootfs/eeprom/parameters*",
                "rootfs/dataman", "rootfs/0/parameters*.bson",
                "rootfs/0/eeprom/parameters*", "rootfs/0/dataman"]
        for p in pats:
            for f in glob.glob(str(root / p)):
                try:
                    os.remove(f)
                except IsADirectoryError:
                    pass
                except FileNotFoundError:
                    pass

    def start(self):
        kill_all()
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
        })
        if self.cfg.headless:
            env["HEADLESS"] = "1"
        self._log_fh = open(self.log_path, "w")
        # -d: daemon mode, no interactive pxh shell (we have no TTY)
        self.proc = subprocess.Popen(
            [str(binary), "-d"], cwd=self.cfg.px4_dir, env=env,
            stdout=self._log_fh, stderr=subprocess.STDOUT,
            start_new_session=True)
        time.sleep(3.0)
        if self.proc.poll() is not None:
            raise RuntimeError(f"PX4 exited immediately; see {self.log_path}")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=8)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        if self._log_fh:
            self._log_fh.close()
        kill_all()

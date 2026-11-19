"""Real-time hot-folder automation runner backed by a warm GPU EnginePool."""

from __future__ import annotations

import json
import shutil
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

from .image_io import INPUT_EXTENSIONS, SUPPORTED_EXTENSIONS
from .inference import EnginePool
from .pipeline import EnhancementConfig, enhance_file

_IGNORED_TEMP_SUFFIXES = (".part", ".tmp", ".crdownload", ".download", ".partial")


@dataclass
class WatchEvent:
    """Single automation event recorded when a file is enhanced or skipped with error."""

    timestamp: float
    status: str  # "enhanced" | "error"
    input_path: str
    output_path: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    device_used: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


class WatchFolderRunner:
    """Watches an input directory, waits for write completion, and enhances new images automatically."""

    def __init__(
        self,
        watch_dir: str | Path,
        output_dir: Optional[str | Path] = None,
        *,
        archive_dir: Optional[str | Path] = None,
        cfg: Optional[EnhancementConfig] = None,
        ext: Optional[str] = None,
        suffix: str = "_azzani",
        recursive: bool = False,
        comparison: bool = False,
        poll_interval: float = 1.0,
        stability_delay: float = 0.35,
        skip_existing: bool = False,
        engine_pool: Optional[EnginePool] = None,
        event_cb: Optional[Callable[[WatchEvent], None]] = None,
        webhook_url: Optional[str] = None,
        webhook_secret: Optional[str] = None,
    ) -> None:
        self.watch_dir = Path(watch_dir).expanduser().resolve()
        self.output_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else (self.watch_dir / "enhanced").resolve()
        )
        self.archive_dir = Path(archive_dir).expanduser().resolve() if archive_dir else None
        self.cfg = cfg or EnhancementConfig(quiet=True)
        self.ext = ext
        self.suffix = suffix
        self.recursive = recursive
        self.comparison = comparison
        self.poll_interval = max(0.2, float(poll_interval))
        self.stability_delay = max(0.0, float(stability_delay))
        self.skip_existing = skip_existing
        self._owns_pool = engine_pool is None
        self.engine_pool = engine_pool or EnginePool(max_models=2)
        self.event_cb = event_cb
        self.webhook_url = webhook_url.strip() if webhook_url else None
        self.webhook_secret = webhook_secret

        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.archive_dir:
            self.archive_dir.mkdir(parents=True, exist_ok=True)

        self._manifest_path = self.output_dir / ".azzani_manifest.json"
        self._manifest: Dict[str, Dict[str, object]] = self._load_manifest()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        self.processed_count = 0
        self.error_count = 0
        self.recent_events: Deque[WatchEvent] = deque(maxlen=50)

        if self.skip_existing:
            self._seed_existing_files()

    def _load_manifest(self) -> Dict[str, Dict[str, object]]:
        if not self._manifest_path.is_file():
            return {}
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def _save_manifest(self) -> None:
        try:
            tmp = self._manifest_path.with_name(self._manifest_path.name + ".tmp")
            tmp.write_text(json.dumps(self._manifest, indent=2), encoding="utf-8")
            tmp.replace(self._manifest_path)
        except Exception:
            pass

    def _is_candidate(self, path: Path) -> bool:
        if not path.is_file():
            return False
        name = path.name
        low = name.lower()
        if name.startswith(".") or low.endswith(_IGNORED_TEMP_SUFFIXES):
            return False
        if path.suffix.lower() not in INPUT_EXTENSIONS:
            return False
        if _is_relative_to(path, self.output_dir):
            return False
        if self.archive_dir and _is_relative_to(path, self.archive_dir):
            return False
        stem = path.stem
        if self.suffix and (stem.endswith(self.suffix) or stem.endswith(f"{self.suffix}_compare")):
            return False
        if stem.endswith("_compare"):
            return False
        return True

    def _list_candidates(self) -> List[Path]:
        it = self.watch_dir.rglob("*") if self.recursive else self.watch_dir.iterdir()
        return sorted(p for p in it if self._is_candidate(p))

    def _seed_existing_files(self) -> None:
        for p in self._list_candidates():
            rel = str(p.relative_to(self.watch_dir))
            st = p.stat()
            self._manifest[rel] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "output": "", "skipped": True}
        self._save_manifest()

    def _target_output_path(self, src: Path) -> Path:
        rel = src.relative_to(self.watch_dir)
        target_ext = f".{self.ext.lstrip('.')}" if self.ext else src.suffix
        if target_ext.lower() not in SUPPORTED_EXTENSIONS:
            target_ext = ".png"
        return (self.output_dir / rel).with_name(f"{src.stem}{self.suffix}{target_ext}")

    def _is_stable(self, src: Path, check_stability: bool) -> Optional[tuple[int, int]]:
        try:
            st1 = src.stat()
        except FileNotFoundError:
            return None
        if st1.st_size <= 0:
            return None
        if not check_stability or self.stability_delay <= 0:
            return st1.st_size, st1.st_mtime_ns
        time.sleep(self.stability_delay)
        try:
            st2 = src.stat()
        except FileNotFoundError:
            return None
        if st1.st_size != st2.st_size or st1.st_mtime_ns != st2.st_mtime_ns:
            return None
        return st2.st_size, st2.st_mtime_ns

    def process_once(self, *, check_stability: bool = False) -> List[WatchEvent]:
        """Scan watch_dir once and enhance any new or updated images synchronously."""
        events: List[WatchEvent] = []
        for src in self._list_candidates():
            if self._stop_event.is_set():
                break
            rel = str(src.relative_to(self.watch_dir))
            st = self._is_stable(src, check_stability=check_stability)
            if st is None:
                continue
            size_b, mtime_ns = st

            prev = self._manifest.get(rel)
            out_path = self._target_output_path(src)
            if prev and prev.get("size") == size_b and prev.get("mtime_ns") == mtime_ns:
                prev_out = prev.get("output")
                if prev.get("skipped") or (isinstance(prev_out, str) and Path(prev_out).is_file()):
                    continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            comp_path = out_path.with_name(f"{out_path.stem}_compare.png") if self.comparison else None

            try:
                res = enhance_file(
                    src,
                    out_path,
                    cfg=self.cfg,
                    comparison_path=comp_path,
                    engine_pool=self.engine_pool,
                )
                self._manifest[rel] = {
                    "size": size_b,
                    "mtime_ns": mtime_ns,
                    "output": str(out_path),
                    "processed_at": time.time(),
                }
                self._save_manifest()

                if self.archive_dir:
                    dest_arch = self.archive_dir / src.relative_to(self.watch_dir)
                    dest_arch.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dest_arch))

                ev = WatchEvent(
                    timestamp=time.time(),
                    status="enhanced",
                    input_path=str(src),
                    output_path=str(out_path),
                    elapsed_seconds=round(res.elapsed_seconds, 3),
                    device_used=res.device_used,
                )
                with self._state_lock:
                    self.processed_count += 1
                    self.recent_events.appendleft(ev)
                events.append(ev)
                if self.webhook_url:
                    from .server import dispatch_webhook

                    dispatch_webhook(
                        self.webhook_url,
                        "watch.enhanced",
                        dict(ev.to_dict()),
                        secret=self.webhook_secret,
                    )
                if self.event_cb:
                    self.event_cb(ev)
            except Exception as exc:
                ev = WatchEvent(
                    timestamp=time.time(),
                    status="error",
                    input_path=str(src),
                    error=str(exc),
                )
                with self._state_lock:
                    self.error_count += 1
                    self.recent_events.appendleft(ev)
                events.append(ev)
                if self.webhook_url:
                    from .server import dispatch_webhook

                    dispatch_webhook(
                        self.webhook_url,
                        "watch.error",
                        dict(ev.to_dict()),
                        secret=self.webhook_secret,
                    )
                if self.event_cb:
                    self.event_cb(ev)
        return events

    def run_forever(self) -> None:
        """Poll continuously until stop() is called or interrupted."""
        try:
            while not self._stop_event.is_set():
                self.process_once(check_stability=True)
                self._stop_event.wait(self.poll_interval)
        finally:
            if self._owns_pool:
                self.engine_pool.clear()

    def start_background(self) -> None:
        """Launch folder watcher in a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run_forever, name="azzani-watch", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the watcher loop to stop and reclaim GPU memory cleanly."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        if self._owns_pool:
            self.engine_pool.clear()

    def status_dict(self) -> Dict[str, object]:
        with self._state_lock:
            return {
                "running": bool(self._thread is not None and self._thread.is_alive()),
                "watch_dir": str(self.watch_dir),
                "output_dir": str(self.output_dir),
                "archive_dir": str(self.archive_dir) if self.archive_dir else None,
                "processed_count": self.processed_count,
                "error_count": self.error_count,
                "recent_events": [e.to_dict() for e in list(self.recent_events)[:10]],
            }

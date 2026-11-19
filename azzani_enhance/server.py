"""Headless REST API queue, HMAC-signed webhook callbacks, and NAS/Docker server runner."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlparse

from .automation import WatchFolderRunner

if TYPE_CHECKING:
    from .web_ui import WebUIState


def dispatch_webhook(
    url: str,
    event: str,
    payload: Dict[str, Any],
    *,
    secret: Optional[str] = None,
    timeout: float = 10.0,
) -> Tuple[bool, Optional[int], Optional[str]]:
    """Send an HTTP JSON POST webhook notification with optional HMAC-SHA256 signature."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False, None, f"Invalid webhook URL scheme or host: '{url}'"

    envelope = {
        "event": event,
        "timestamp": round(time.time(), 3),
        "data": payload,
    }
    raw = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "AzzaniEnhance-Webhook/1.0",
        "X-Azzani-Event": event,
    }
    if secret:
        sig = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
        headers["X-Azzani-Signature"] = f"sha256={sig}"

    req = urllib.request.Request(url.strip(), data=raw, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            status = int(getattr(resp, "status", 200))
            return 200 <= status < 300, status, None
    except urllib.error.HTTPError as err:
        return False, int(err.code), f"HTTP {err.code}: {err.reason}"
    except Exception as exc:
        return False, None, str(exc)


class AsyncJobManager:
    """Thread-safe FIFO background job queue sharing the server's warm EnginePool."""

    def __init__(
        self,
        worker_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        *,
        max_jobs: int = 64,
    ) -> None:
        self._worker_fn = worker_fn
        self._max_jobs = max(8, int(max_jobs))
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._order: list[str] = []
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=self._max_jobs)
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run_loop, name="azzani-api-worker", daemon=True
        )
        self._thread.start()

    def enqueue(self, job_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self._stopped:
                raise RuntimeError("Job manager is shutting down.")
            record = {
                "job_id": job_id,
                "status": "queued",
                "created_at": round(time.time(), 3),
                "started_at": None,
                "finished_at": None,
                "filename": spec.get("filename"),
                "before_url": None,
                "preview_url": None,
                "result_url": None,
                "download_name": None,
                "summary": None,
                "histogram": None,
                "webhook_delivered": None,
                "webhook_error": None,
                "error": None,
                "_spec": spec,
            }
            self._jobs[job_id] = record
            self._order.append(job_id)
            while len(self._order) > self._max_jobs:
                old_id = self._order.pop(0)
                old = self._jobs.get(old_id)
                if old and old.get("status") in {"completed", "failed"}:
                    self._jobs.pop(old_id, None)
        try:
            self._queue.put_nowait(job_id)
        except queue.Full as err:
            with self._lock:
                self._jobs.pop(job_id, None)
            raise OverflowError("Async job queue is full; try again shortly.") from err
        return self.get_public_status(job_id) or {}

    def get_public_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return None
            return {k: v for k, v in rec.items() if not k.startswith("_")}

    def queue_depth(self) -> int:
        with self._lock:
            return sum(
                1 for r in self._jobs.values() if r.get("status") in {"queued", "running"}
            )

    def _run_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                self._queue.task_done()
                break
            with self._lock:
                rec = self._jobs.get(job_id)
                if rec is None:
                    self._queue.task_done()
                    continue
                rec["status"] = "running"
                rec["started_at"] = round(time.time(), 3)
                spec = dict(rec.get("_spec") or {})
                rec.pop("_spec", None)

            webhook_url = spec.get("webhook_url")
            webhook_secret = spec.get("webhook_secret")
            try:
                out = self._worker_fn(spec)
                with self._lock:
                    rec.update(out)
                    rec["status"] = "completed"
                    rec["finished_at"] = round(time.time(), 3)
                event_name = "enhancement.completed"
                payload = self.get_public_status(job_id) or {}
            except Exception as exc:
                with self._lock:
                    rec["status"] = "failed"
                    rec["error"] = str(exc)
                    rec["finished_at"] = round(time.time(), 3)
                event_name = "enhancement.failed"
                payload = self.get_public_status(job_id) or {}

            if isinstance(webhook_url, str) and webhook_url.strip():
                ok, _code, err_msg = dispatch_webhook(
                    webhook_url, event_name, payload, secret=webhook_secret
                )
                with self._lock:
                    rec["webhook_delivered"] = ok
                    rec["webhook_error"] = err_msg

            self._queue.task_done()

    def shutdown(self) -> None:
        with self._lock:
            self._stopped = True
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


def launch_api_server(
    *,
    host: str = "0.0.0.0",
    port: int = 8765,
    api_key: Optional[str] = None,
    models_dir: Optional[Path] = None,
    watch_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    webhook_url: Optional[str] = None,
    webhook_secret: Optional[str] = None,
) -> int:
    """Run headless REST API server (+ optional background hot-folder automation)."""
    from .pipeline import EnhancementConfig
    from .web_ui import create_web_server

    resolved_key = api_key if api_key is not None else (os.environ.get("AZZANI_API_KEY") or None)
    try:
        server, state = create_web_server(
            host=host,
            port=port,
            models_dir=models_dir,
            api_key=resolved_key,
        )
    except OSError:
        server, state = create_web_server(
            host=host,
            port=0,
            models_dir=models_dir,
            api_key=resolved_key,
        )

    if watch_dir is not None:
        wdir = Path(watch_dir).expanduser().resolve()
        odir = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else (wdir / "enhanced").resolve()
        )
        with state.watch_lock:
            state.watcher = WatchFolderRunner(
                watch_dir=wdir,
                output_dir=odir,
                cfg=EnhancementConfig(models_dir=models_dir, quiet=True),
                engine_pool=state.engine_pool,
                webhook_url=webhook_url,
                webhook_secret=webhook_secret,
            )
            state.watcher.start_background()

    bound_host, bound_port = server.server_address[:2]
    print(f"AZZANI Headless REST API listening on http://{bound_host}:{bound_port}")
    print(f"  Healthcheck : http://{bound_host}:{bound_port}/healthz")
    print(f"  Sync API    : POST http://{bound_host}:{bound_port}/api/enhance")
    print(f"  Async Queue : POST http://{bound_host}:{bound_port}/api/enhance/async")
    if resolved_key:
        print("  API Auth    : Enabled (X-API-Key / Bearer token required on /api/*)")
    if watch_dir is not None:
        print(f"  Watch Folder: {Path(watch_dir).expanduser().resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down AZZANI API server cleanly...")
    finally:
        server.server_close()
        state.cleanup()
    return 0

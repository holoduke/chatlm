import psutil

# Cached ollama Process handle (re-resolved if it dies).
_ollama_proc: psutil.Process | None = None

# Prime the non-blocking cpu_percent() counters so later calls return
# meaningful deltas without a blocking interval.
psutil.cpu_percent(interval=None)


def _resolve_ollama_process() -> psutil.Process | None:
    global _ollama_proc
    if _ollama_proc is not None:
        try:
            if _ollama_proc.is_running():
                return _ollama_proc
        except psutil.NoSuchProcess:
            pass
        _ollama_proc = None
    for proc in psutil.process_iter(["name", "cmdline"]):
        name = (proc.info.get("name") or "").lower()
        cmdline = " ".join(proc.info.get("cmdline") or []).lower()
        if (name == "ollama" or cmdline.endswith("ollama serve") or " ollama serve" in cmdline) and "serve" in cmdline:
            _ollama_proc = proc
            try:
                proc.cpu_percent(interval=None)  # prime per-proc counter
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            return proc
    return None


def collect() -> dict:
    vm = psutil.virtual_memory()
    stats: dict = {
        "system": {
            "memory_total_gb": round(vm.total / 1024**3, 2),
            "memory_used_gb": round(vm.used / 1024**3, 2),
            "memory_percent": vm.percent,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_count": psutil.cpu_count(logical=True),
        },
        "ollama": None,
    }
    proc = _resolve_ollama_process()
    if proc is not None:
        try:
            with proc.oneshot():
                stats["ollama"] = {
                    "pid": proc.pid,
                    "rss_mb": round(proc.memory_info().rss / 1024**2, 1),
                    "cpu_percent": round(proc.cpu_percent(interval=None), 1),
                    "threads": proc.num_threads(),
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return stats

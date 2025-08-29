#!/usr/bin/env python3
import argparse, json, os, signal, sys, time, datetime as dt

try:
    import psutil, pynvml
except Exception as e:
    sys.stderr.write(f"[FATAL] Import failed: {e}\n")
    sys.exit(2)

RUNNING = True
def _stop(*_):
    global RUNNING
    RUNNING = False

def now():
    return dt.datetime.now().isoformat(timespec="seconds")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="auto", help='"auto" or 0-based visible index')
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--outfile", required=True)
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    pynvml.nvmlInit()
    try:
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            sys.stderr.write("[FATAL] No GPUs visible.\n"); return 3
        gid = 0 if str(args.gpu) == "auto" else int(args.gpu)
        if gid < 0 or gid >= count:
            sys.stderr.write(f"[FATAL] Invalid GPU {gid}; visible 0..{count-1}\n"); return 4
        h = pynvml.nvmlDeviceGetHandleByIndex(gid)
        name = pynvml.nvmlDeviceGetName(h).decode() if isinstance(pynvml.nvmlDeviceGetName(h), bytes) else str(pynvml.nvmlDeviceGetName(h))

        os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
        with open(args.outfile, "a", buffering=1) as f:
            sys.stderr.write(f"[INFO] SystemMonitor started @ {args.outfile} every {args.interval}s (GPU {gid} {name})\n")
            psutil.cpu_percent(None)  # prime
            while RUNNING:
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                row = {
                    "timestamp": now(),
                    "gpu_index": gid,
                    "gpu_name": name,
                    "power_watts": round(power, 3),
                    "memory_used_MB": round(mem.used/(1024**2), 3),
                    "gpu_utilization_percent": int(util.gpu),
                    "memory_utilization_percent": int(util.memory),
                    "temperature_C": int(temp),
                    "cpu_utilization_percent": float(psutil.cpu_percent(None)),
                }
                f.write(json.dumps(row) + "\n")
                f.flush(); os.fsync(f.fileno())
                time.sleep(args.interval)
    finally:
        try: pynvml.nvmlShutdown()
        except: pass

if __name__ == "__main__":
    sys.exit(main())

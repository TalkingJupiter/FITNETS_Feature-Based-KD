import pynvml
import time
import csv
import psutil
from datetime import datetime
import argparse

class PowerMonitor:
    def __init__(self, gpu_index=0, interval=0.5, outfile="power_log.csv"):
        self.gpu_index = gpu_index
        self.interval = interval
        self.outfile = outfile
        self.running = False

    def __enter__(self):
        pynvml.nvmlInit()
        self.device = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        self.gpu_name = pynvml.nvmlDeviceGetName(self.device).decode("utf-8")
        self.start_time = time.time()
        self.logs = []
        self.running = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        pynvml.nvmlShutdown()
        if self.logs:
            with open(self.outfile, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.logs[0].keys())
                writer.writeheader()
                for row in self.logs:
                    writer.writerow(row)
            print(f"[Monitor] Saved telemetry log to: {self.outfile}")

    def log_once(self):
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.device)
        util = pynvml.nvmlDeviceGetUtilizationRates(self.device)
        temp = pynvml.nvmlDeviceGetTemperature(self.device, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(self.device) / 1000.0  # mW → W
        cpu_util = psutil.cpu_percent(interval=None)

        record = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_sec": round(time.time() - self.start_time, 3),
            "gpu_index": self.gpu_index,
            "gpu_name": self.gpu_name,
            "power_watts": round(power, 3),
            "memory_used_MB": round(mem_info.used / 1024**2, 2),
            "memory_total_MB": round(mem_info.total / 1024**2, 2),
            "gpu_utilization_percent": util.gpu,
            "memory_utilization_percent": util.memory,
            "temperature_C": temp,
            "cpu_utilization_percent": round(cpu_util, 2),
        }

        self.logs.append(record)

    def start(self):
        print(f"[Monitor] Monitoring GPU {self.gpu_index} every {self.interval}s. Output: {self.outfile}")
        try:
            while self.running:
                self.log_once()
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("[Monitor] Interrupted by user.")

# ------------------ CLI ENTRY POINT ------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Power Telemetry Monitor")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to monitor")
    parser.add_argument("--interval", type=float, default=0.5, help="Sampling interval (seconds)")
    parser.add_argument("--log_path", type=str, default="power_log.csv", help="CSV output path")
    args = parser.parse_args()

    monitor = PowerMonitor(
        gpu_index=args.gpu,
        interval=args.interval,
        outfile=args.log_path
    )

    with monitor:
        monitor.start()

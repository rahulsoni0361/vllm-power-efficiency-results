import subprocess
import time
import re
import threading
import json
from datetime import datetime

# --- Config ---
SERVER_IP = "192.168.1.136"
SSH_CMD = f"ssh ray@{SERVER_IP}"
STRESS_TEST_PATH = "~/vllm_mcp/stress_test.py"
PYTHON_BIN = "~/vllm_mcp/miniconda/envs/vllm_env/bin/python -u"
SUDO_PASS = "YOUR_PASSWORD"  # Replace with your actual sudo password

def run_remote(cmd, input_str=None):
    """Run a command on the remote server and return stdout."""
    full_cmd = f"{SSH_CMD} \"{cmd}\""
    process = subprocess.Popen(full_cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if input_str:
        process.stdin.write(input_str)
        process.stdin.close()
    
    lines = []
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(f"  [Remote] {line.strip()}")
            lines.append(line)
    
    return "".join(lines), ""

def set_power_limit(limit_watts):
    print(f"--- Setting Power Limit to {limit_watts}W ---")
    cmd = f"echo '{SUDO_PASS}' | sudo -S nvidia-smi -pl {limit_watts}"
    run_remote(cmd)

def run_remote_silent(cmd):
    """Run a command on the remote server silently."""
    full_cmd = f"{SSH_CMD} \"{cmd}\""
    return subprocess.check_output(full_cmd, shell=True, text=True)

def monitor_gpu(stop_event, stats_list):
    """Background thread to monitor GPU metrics."""
    while not stop_event.is_set():
        # Query temp and power draw
        try:
            stdout = run_remote_silent("nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits")
            temp, power = map(float, stdout.strip().split(','))
            stats_list.append({"temp": temp, "power": power, "time": time.time()})
        except:
            pass
        time.sleep(2)

def run_experiment(watt_limit):
    set_power_limit(watt_limit)
    print(f"--- Starting Stress Test at {watt_limit}W ---")
    
    metrics = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_gpu, args=(stop_event, metrics))
    monitor_thread.start()
    
    # Run the stress test
    # Note: We need to use -u for unbuffered output to potentially capture progress, 
    # but here we mainly care about the summary at the end.
    stdout, stderr = run_remote(f"{PYTHON_BIN} {STRESS_TEST_PATH}")
    
    stop_event.set()
    monitor_thread.join()
    
    # Parse stats from stdout
    # Example line: Throughput:        0.12 req/s | 61.2 tokens/s
    results = {
        "limit": watt_limit,
        "tps": 0.0,
        "avg_temp": 0.0,
        "max_temp": 0.0,
        "avg_power": 0.0,
        "duration": 0
    }
    
    tps_match = re.search(r"\|\s*([\d\.]+)\s*tokens/s", stdout)
    if tps_match:
        results["tps"] = float(tps_match.group(1))
    
    duration_match = re.search(r"Duration:\s*(\d+)s", stdout)
    if duration_match:
        results["duration"] = int(duration_match.group(1))
        
    if metrics:
        temps = [m["temp"] for m in metrics]
        powers = [m["power"] for m in metrics]
        results["avg_temp"] = sum(temps) / len(temps)
        results["max_temp"] = max(temps)
        results["avg_power"] = sum(powers) / len(powers)
        
    return results

def main():
    print("Starting Power vs Performance Experiment...")
    
    # Run 1: 350W
    res_350 = run_experiment(350)
    print(f"350W Run Complete: {res_350['tps']} tokens/s, Avg Temp: {res_350['avg_temp']:.1f}C")
    
    print("\nWaiting 30s for cool-down...")
    time.sleep(30)
    
    # Run 2: 300W
    res_300 = run_experiment(300)
    print(f"300W Run Complete: {res_300['tps']} tokens/s, Avg Temp: {res_300['avg_temp']:.1f}C")
    
    # Reset to default
    set_power_limit(350)
    
    # Print Table
    print("\n" + "="*80)
    print(f"{'Limit':<10} | {'Throughput':<15} | {'Avg Temp':<10} | {'Max Temp':<10} | {'Avg Power':<12}")
    print("-" * 80)
    for r in [res_350, res_300]:
        print(f"{r['limit']:<10} | {r['tps']:<15.1f} | {r['avg_temp']:<10.1f} | {r['max_temp']:<10.1f} | {r['avg_power']:<12.1f}")
    print("="*80)

    # Save to file
    with open("experiment_results.json", "w") as f:
        json.dump([res_350, res_300], f, indent=2)

if __name__ == "__main__":
    main()

"""
vLLM Stress Test - DSP Tutorial Generator
Hammers the vLLM API with concurrent requests for several minutes.
Monitor GPU with: watch -n 1 nvidia-smi
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import time
import threading
import random
from openai import OpenAI
from datetime import datetime

# --- Config ---
BASE_URL = "http://192.168.1.136:8000/v1"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DURATION_SECONDS = 2 * 60  # 2 minutes
CONCURRENT_WORKERS = 1      # Parallel request threads
MAX_TOKENS = 512             # Tokens per response

DSP_TOPICS = [
    "Explain the Fast Fourier Transform (FFT) and why it's faster than the DFT.",
    "Write a tutorial on FIR filter design with a windowing method.",
    "Explain the Nyquist-Shannon sampling theorem with a real-world example.",
    "Write a tutorial on IIR filters vs FIR filters: pros and cons.",
    "Explain convolution in the context of digital signal processing.",
    "Write a tutorial on the Z-transform and its applications in DSP.",
    "Explain windowing functions: Hann, Hamming, Blackman — when to use each.",
    "Write a tutorial on Power Spectral Density (PSD) estimation.",
    "Explain decimation and interpolation in multirate signal processing.",
    "Write a tutorial on the Short-Time Fourier Transform (STFT) and spectrograms.",
    "Explain digital filter implementation in Python using scipy.signal.",
    "Write a tutorial on adaptive filtering and the LMS algorithm.",
    "Explain the Discrete Cosine Transform (DCT) and its use in audio compression.",
    "Write a tutorial on using FFT to detect frequency components in a noisy signal.",
    "Explain the Hilbert transform and its role in computing the analytic signal.",
    "Write a tutorial on audio equalization using digital filters.",
    "Explain phase response and group delay in digital filters.",
    "Write a tutorial on wavelet transforms as an alternative to STFT.",
    "Explain quantization noise and its impact in digital signal processing.",
    "Write a tutorial on matched filtering for signal detection in noise.",
]

# --- Shared Statistics ---
lock = threading.Lock()
stats = {
    "requests_sent": 0,
    "requests_done": 0,
    "requests_failed": 0,
    "total_tokens": 0,
    "start_time": None,
}

client = OpenAI(base_url=BASE_URL, api_key="none")


def run_request(worker_id: int):
    topic = random.choice(DSP_TOPICS)
    prompt = f"You are a DSP expert. {topic} Be thorough and include code examples where relevant."
    
    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        elapsed = time.time() - t0
        usage = response.usage
        content_preview = response.choices[0].message.content[:80].replace("\n", " ")

        with lock:
            stats["requests_done"] += 1
            stats["total_tokens"] += usage.completion_tokens if usage else 0

        print(
            f"[Worker {worker_id}] [OK] Done in {elapsed:.1f}s | "
            f"{usage.completion_tokens if usage else '?'} tokens | "
            f'Topic: "{topic[:50]}..." | '
            f'Reply: "{content_preview}..."'
        )

    except Exception as e:
        with lock:
            stats["requests_failed"] += 1
        print(f"[Worker {worker_id}] [ERR] Error: {e}")


def worker_loop(worker_id: int, stop_event: threading.Event):
    """Each worker continuously sends requests until the stop event is set."""
    while not stop_event.is_set():
        with lock:
            stats["requests_sent"] += 1
        run_request(worker_id)


def print_summary():
    elapsed = time.time() - stats["start_time"]
    rps = stats["requests_done"] / elapsed if elapsed > 0 else 0
    tps = stats["total_tokens"] / elapsed if elapsed > 0 else 0
    print("\n" + "=" * 70)
    print("  STRESS TEST COMPLETE")
    print("=" * 70)
    print(f"  Duration:          {elapsed:.0f}s")
    print(f"  Requests sent:     {stats['requests_sent']}")
    print(f"  Requests completed:{stats['requests_done']}")
    print(f"  Requests failed:   {stats['requests_failed']}")
    print(f"  Total tokens out:  {stats['total_tokens']}")
    print(f"  Throughput:        {rps:.2f} req/s | {tps:.1f} tokens/s")
    print("=" * 70)


def main():
    print("=" * 70)
    print(f"  vLLM DSP Tutorial Stress Test")
    print(f"  Model:    {MODEL}")
    print(f"  Workers:  {CONCURRENT_WORKERS}")
    print(f"  Duration: {DURATION_SECONDS}s ({DURATION_SECONDS//60}m)")
    print(f"  Monitor GPU with: watch -n 1 nvidia-smi")
    print("=" * 70)
    print(f"  Starting at {datetime.now().strftime('%H:%M:%S')}...\n")

    stats["start_time"] = time.time()
    stop_event = threading.Event()

    threads = [
        threading.Thread(target=worker_loop, args=(i + 1, stop_event), daemon=True)
        for i in range(CONCURRENT_WORKERS)
    ]
    for t in threads:
        t.start()

    try:
        time.sleep(DURATION_SECONDS)
    except KeyboardInterrupt:
        print("\n  Interrupted by user.")

    print(f"\n  Stopping workers...")
    stop_event.set()
    for t in threads:
        t.join(timeout=60)

    print_summary()


if __name__ == "__main__":
    main()

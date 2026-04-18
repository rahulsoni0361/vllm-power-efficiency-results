# RTX 3090 Power Efficiency Experiment: vLLM & Qwen2.5-VL

This repository contains the results of a controlled experiment comparing the performance and thermal efficiency of an NVIDIA RTX 3090 when running local AI inference (vLLM) at different power limits.

## The Experiment
- **Model**: `Qwen/Qwen2.5-VL-7B-Instruct`
- **Engine**: `vLLM v0.19.0`
- **Hardware**: NVIDIA RTX 3090 (24GB VRAM)
- **Workload**: 2-minute sustained stress test with concurrent DSP tutorial generation.

## Results Summary

| Power Limit | Throughput (TPS) | Avg Temp | Max Temp | Avg Power Draw |
| :--- | :--- | :--- | :--- | :--- |
| **350W** (Stock) | **47.5 tokens/s** | 75.1°C | 76.0°C | 300.8W |
| **300W** (Limited) | **47.6 tokens/s** | 76.0°C | 77.0°C | 297.4W |

## Key Takeaways
1. **Diminishing Returns**: For the 7B variant of Qwen2.5-VL, a single worker does not saturate the 350W power budget.
2. **Efficiency**: Reducing the power limit to 300W had **zero impact** on tokens per second.
3. **Hardware Health**: Limiting power is a "free" way to reduce transient spikes and ensure VRAM longevity without sacrificing speed.

## Reproducing the Results
The orchestration script `power_experiment.py` and the `stress_test.py` used for this benchmark are included in this repository.

### Requirements
- Linux Server with NVIDIA GPU
- `vLLM` installed in a conda environment
- `nvidia-smi` access

### Running
```bash
python power_experiment.py
```

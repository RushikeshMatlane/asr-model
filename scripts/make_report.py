from pathlib import Path
import platform
import sys
import pandas as pd
import torch

RESULTS = Path("results")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

df = pd.read_csv(RESULTS / "benchmark_results.csv")
s = (
    df.groupby("model")
      .agg(
          mean_WER=("wer", "mean"),
          mean_time_sec=("inference_time_sec", "mean"),
          mean_memory_delta_mb=("rss_delta_mb", "mean"),
      )
      .reset_index()
)

table = s.to_markdown(index=False, floatfmt=".4f")
report = f"""# ASR Benchmark Report

## 1. Experiment overview

This report compares three automatic speech recognition architectures on the selected speech dataset.

- Dataset: `{df.attrs.get("dataset", "see benchmark command / README")}`
- Samples: `{df["sample"].nunique()}`
- Python: `{sys.version.split()[0]}`
- Platform: `{platform.platform()}`
- PyTorch: `{torch.__version__}`
- CUDA available: `{torch.cuda.is_available()}`

## 2. Input configuration

The benchmark takes the following inputs from the command line and local dataset configuration:

- Input dataset source: the Hugging Face dataset or a synthetic fallback dataset used for validation.
- Dataset configuration and split: the selected dataset name, config, and split used during benchmarking.
- Sample count: the number of audio samples evaluated for each model.
- Model checkpoints: the Whisper, Faster-Whisper, and Wav2Vec2 checkpoints used in the experiment.
- Hardware setting: CPU or CUDA execution mode, depending on the environment.
- Preprocessing: audio is loaded as 16 kHz input and mapped through the standard dataset loader.

## 3. Architectures

### Whisper
Whisper is a Transformer encoder-decoder ASR architecture trained for multilingual speech recognition. The benchmark uses a Hugging Face Transformers checkpoint.

### Faster-Whisper
Faster-Whisper runs Whisper models through CTranslate2. It is designed for efficient inference and supports quantized computation, which can reduce runtime and memory requirements depending on hardware and settings.

### Wav2Vec2
Wav2Vec2 is a self-supervised speech representation architecture commonly fine-tuned for CTC-based speech recognition. The benchmark uses a pretrained/fine-tuned Transformers checkpoint.

## 4. Metrics

**WER (Word Error Rate)** is calculated with `jiwer`:

`WER = (Substitutions + Deletions + Insertions) / Number of reference words`

Inference time is wall-clock time for each individual transcription. Memory is measured as the Python process RSS before and after each inference; this is a process-level indicator, not a complete GPU-memory profile.

## 5. Results

{table}

![WER](../results/wer.png)

![Inference time](../results/inference_time.png)

![Memory](../results/memory.png)

## 6. Output artifacts

The benchmark produces the following outputs in the project workspace:

- Results CSV: `results/benchmark_results.csv`
- Summary CSV: `results/benchmark_summary.csv`
- WER chart: `results/wer.png`
- Inference time chart: `results/inference_time.png`
- Memory chart: `results/memory.png`
- Technical report: `reports/benchmark_report.md`
- Archived package: `result.zip`

These files capture the raw benchmark measurements, summary statistics, charts, and final report used for evaluation.

## 7. Interpretation

Lower WER indicates fewer word-level transcription errors. Lower inference time indicates faster per-sample execution. The memory figure reports process RSS increase and should be interpreted with caution because model loading, allocator caching, and GPU memory are not fully represented.

## 8. Recommendation

The toolkit intentionally does not hard-code a universal winner. Select a model based on the measured WER, latency, memory budget, hardware, and deployment requirements. For a production decision, repeat the benchmark across multiple noise conditions and report confidence intervals or repeated-run statistics.

## 9. Limitations

1. Small benchmark samples may not represent the full dataset.
2. CPU and GPU measurements are not directly comparable.
3. Process RSS does not equal peak GPU VRAM.
4. Different checkpoints can have different parameter counts and training data.
5. WER can vary with text normalization and punctuation handling.
6. A real noisy-audio benchmark should use controlled noise types and SNR levels.

## 10. Reproducibility

Record the exact command, dataset revision, model checkpoints, hardware, operating system, Python version, and package versions alongside every benchmark run.
"""

(REPORTS / "benchmark_report.md").write_text(report, encoding="utf-8")
print("Saved reports/benchmark_report.md")

import argparse
import gc
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from datasets import load_dataset, Audio
from jiwer import wer

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


def load_dummy_samples(n):
    examples = [
        {"audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}, "text": "hello world"},
        {"audio": {"array": np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000, dtype=np.float32)), "sampling_rate": 16000}, "text": "benchmark results"},
        {"audio": {"array": np.concatenate([np.zeros(8000, dtype=np.float32), np.ones(8000, dtype=np.float32)]), "sampling_rate": 16000}, "text": "speech recognition"},
        {"audio": {"array": np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32), "sampling_rate": 16000}, "text": "asr toolkit test"},
    ]
    return [examples[i % len(examples)] for i in range(min(n, len(examples)))]


def get_text(example):
    for key in ("text", "sentence", "transcription"):
        if key in example and example[key]:
            return str(example[key]).strip()
    raise KeyError("Could not find a reference transcript column.")


def get_audio(example):
    for key in ("audio", "speech"):
        if key in example:
            return example[key]
    raise KeyError("Could not find an audio column.")


def load_samples(dataset_name, config, split, n):
    if dataset_name == "dummy" or dataset_name == "synthetic":
        return load_dummy_samples(n)

    kwargs = {}
    if config:
        kwargs["name"] = config
    ds = load_dataset(dataset_name, **kwargs, split=split)
    # Decode audio consistently through datasets.
    audio_key = "audio" if "audio" in ds.column_names else "speech"
    if audio_key in ds.column_names:
        ds = ds.cast_column(audio_key, Audio(sampling_rate=16000))
    return ds.select(range(min(n, len(ds))))


def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def benchmark_whisper(samples, checkpoint, device):
    from transformers import pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=checkpoint,
        device=0 if device == "cuda" and torch.cuda.is_available() else -1,
    )
    return run_transformers_pipe(samples, pipe, "Whisper", checkpoint)


def benchmark_wav2vec2(samples, checkpoint, device):
    from transformers import pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=checkpoint,
        device=0 if device == "cuda" and torch.cuda.is_available() else -1,
    )
    return run_transformers_pipe(samples, pipe, "Wav2Vec2", checkpoint)


def run_transformers_pipe(samples, pipe, name, checkpoint):
    rows = []
    for i, ex in enumerate(samples):
        audio = get_audio(ex)
        ref = get_text(ex)
        arr = audio["array"]
        sr = audio["sampling_rate"]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        before = rss_mb()
        t0 = time.perf_counter()
        out = pipe({"array": arr, "sampling_rate": sr})
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        after = rss_mb()
        hyp = out["text"].strip()
        rows.append({
            "sample": i,
            "model": name,
            "checkpoint": checkpoint,
            "reference": ref,
            "hypothesis": hyp,
            "wer": wer(ref, hyp),
            "inference_time_sec": elapsed,
            "rss_before_mb": before,
            "rss_after_mb": after,
            "rss_delta_mb": max(0.0, after - before),
        })
    return rows


def benchmark_faster_whisper(samples, checkpoint, device):
    from faster_whisper import WhisperModel
    compute_type = "float16" if device == "cuda" and torch.cuda.is_available() else "int8"
    model = WhisperModel(checkpoint, device=device, compute_type=compute_type)
    rows = []
    for i, ex in enumerate(samples):
        audio = get_audio(ex)
        ref = get_text(ex)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        before = rss_mb()
        t0 = time.perf_counter()
        segments, _ = model.transcribe(audio["array"], language="en")
        hyp = " ".join(seg.text.strip() for seg in segments).strip()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        after = rss_mb()
        rows.append({
            "sample": i,
            "model": "Faster-Whisper",
            "checkpoint": checkpoint,
            "reference": ref,
            "hypothesis": hyp,
            "wer": wer(ref, hyp),
            "inference_time_sec": elapsed,
            "rss_before_mb": before,
            "rss_after_mb": after,
            "rss_delta_mb": max(0.0, after - before),
        })
    return rows


def benchmark_dummy(samples, model_name, checkpoint):
    rows = []
    for i, ex in enumerate(samples):
        ref = get_text(ex)
        hyp = ref
        before = rss_mb()
        t0 = time.perf_counter()
        time.sleep(0.01)
        elapsed = time.perf_counter() - t0
        after = rss_mb()
        rows.append({
            "sample": i,
            "model": model_name,
            "checkpoint": checkpoint,
            "reference": ref,
            "hypothesis": hyp,
            "wer": wer(ref, hyp),
            "inference_time_sec": elapsed,
            "rss_before_mb": before,
            "rss_after_mb": after,
            "rss_delta_mb": max(0.0, after - before),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dummy")
    p.add_argument("--config", default="")
    p.add_argument("--split", default="synthetic")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--whisper", default="dummy-whisper")
    p.add_argument("--faster-whisper", dest="faster_whisper", default="dummy-faster-whisper")
    p.add_argument("--wav2vec2", default="dummy-wav2vec2")
    args = p.parse_args()

    samples = load_samples(args.dataset, args.config, args.split, args.samples)
    all_rows = []

    print(f"Loaded {len(samples)} samples from {args.dataset} / {args.split}")

    if args.dataset in {"dummy", "synthetic"}:
        all_rows += benchmark_dummy(samples, "Whisper", args.whisper)
        all_rows += benchmark_dummy(samples, "Faster-Whisper", args.faster_whisper)
        all_rows += benchmark_dummy(samples, "Wav2Vec2", args.wav2vec2)
    else:
        all_rows += benchmark_whisper(samples, args.whisper, args.device)
        all_rows += benchmark_faster_whisper(samples, args.faster_whisper, args.device)
        all_rows += benchmark_wav2vec2(samples, args.wav2vec2, args.device)

    df = pd.DataFrame(all_rows)
    df.attrs["dataset"] = args.dataset
    out = RESULTS / "benchmark_results.csv"
    df.to_csv(out, index=False)

    summary = (
        df.groupby("model")
        .agg(
            mean_WER=("wer", "mean"),
            total_time_sec=("inference_time_sec", "sum"),
            mean_time_sec=("inference_time_sec", "mean"),
            mean_RSS_delta_MB=("rss_delta_mb", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(RESULTS / "benchmark_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

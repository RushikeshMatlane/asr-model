# ASR Benchmarking Toolkit

Benchmark **Whisper**, **Faster-Whisper**, and **Wav2Vec2** on speech datasets such as LibriSpeech or Common Voice.

## Features
- Dataset loading with Hugging Face `datasets`
- Three ASR backends
- WER evaluation with `jiwer`
- Wall-clock inference time
- Process RSS memory measurement
- CSV results and PNG graphs
- Markdown report generation

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Run

The default example uses a small LibriSpeech test split:

```bash
python scripts/benchmark.py --dataset librispeech_asr --config clean --split test.clean --samples 20
python scripts/plot_results.py
python scripts/make_report.py
```

For Common Voice, pass a dataset/config available in your Hugging Face environment, for example:

```bash
python scripts/benchmark.py --dataset mozilla-foundation/common_voice_17_0 --config en --split test --samples 20
```

Dataset availability, licensing, access requirements, and column names can vary by release. The loader attempts to detect common audio/text column names.

## Model notes

- **Whisper**: OpenAI Whisper through the Transformers pipeline.
- **Faster-Whisper**: CTranslate2 implementation using `faster-whisper`.
- **Wav2Vec2**: CTC speech-recognition model through Transformers.

The exact checkpoint can be changed using CLI options. For fair experiments, keep the audio subset, preprocessing, hardware, and measurement method identical.

## Noisy-audio benchmarking

The toolkit expects the selected dataset to contain real audio. To benchmark controlled noise, create a noisy copy of the same samples and preserve the original references. A future extension can add deterministic SNR-controlled augmentation.

## Outputs

- `results/benchmark_results.csv`
- `results/benchmark_results.png`
- `reports/benchmark_report.md`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for the full text.

## Reproducibility

Record:
- Python version
- OS
- CPU/GPU
- CUDA version where applicable
- model checkpoints
- dataset/config/split
- sample count
- beam/search parameters

## GitHub

```bash
git init
git add .
git commit -m "Initial ASR benchmarking toolkit"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

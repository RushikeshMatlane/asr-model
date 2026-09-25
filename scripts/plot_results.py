from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = Path("results")
df = pd.read_csv(RESULTS / "benchmark_results.csv")

summary = (
    df.groupby("model")
      .agg(WER=("wer", "mean"),
           InferenceTime=("inference_time_sec", "mean"),
           MemoryDeltaMB=("rss_delta_mb", "mean"))
      .reset_index()
)

plt.figure()
plt.bar(summary["model"], summary["WER"])
plt.ylabel("Mean WER")
plt.title("ASR Word Error Rate")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(RESULTS / "wer.png", dpi=180)
plt.close()

plt.figure()
plt.bar(summary["model"], summary["InferenceTime"])
plt.ylabel("Mean inference time (s)")
plt.title("ASR Inference Time")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(RESULTS / "inference_time.png", dpi=180)
plt.close()

plt.figure()
plt.bar(summary["model"], summary["MemoryDeltaMB"])
plt.ylabel("Mean RSS increase (MB)")
plt.title("Process Memory Increase")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(RESULTS / "memory.png", dpi=180)
plt.close()

print("Saved PNG graphs in results/")

import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


LOG_FILES = {
    "Transformer 128/2": "log/Transformer_residual_128_2_100in_100out.txt",
    "Transformer 256/4": "log/Transformer_residual_256_4_100in_100out.txt",
    "Transformer 512/8": "log/Transformer_residual_512_8_100in_100out.txt",
}

MIN_STEP = 2000
ROLLING_WINDOW = 20


def parse_step_loss(text, model_name):
    pattern = re.compile(r"step\s+(\d+);\s+step_loss:\s+([0-9.]+)")
    rows = []

    for step, loss in pattern.findall(text):
        rows.append({
            "model": model_name,
            "step": int(step),
            "step_loss": float(loss),
        })

    return pd.DataFrame(rows)


def parse_eval_loss(text, model_name):
    pattern = re.compile(
        r"Global step:\s+(\d+).*?"
        r"Train loss avg:\s+([0-9.]+).*?"
        r"Val loss:\s+([0-9.]+).*?"
        r"srnn loss:\s+([0-9.]+)",
        re.DOTALL
    )

    rows = []

    for step, train_loss, val_loss, srnn_loss in pattern.findall(text):
        rows.append({
            "model": model_name,
            "step": int(step),
            "train_loss_avg": float(train_loss),
            "val_loss": float(val_loss),
            "srnn_loss": float(srnn_loss),
        })

    return pd.DataFrame(rows)


def load_logs(log_files):
    step_frames = []
    eval_frames = []

    for model_name, file_path in log_files.items():
        path = Path(file_path)

        if not path.exists():
            print(f"Missing file: {path}")
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")

        step_frames.append(parse_step_loss(text, model_name))
        eval_frames.append(parse_eval_loss(text, model_name))

    step_df = pd.concat(step_frames, ignore_index=True)
    eval_df = pd.concat(eval_frames, ignore_index=True)

    return step_df, eval_df


def plot_step_loss(step_df):
    plt.figure(figsize=(10, 6))

    for model_name, df in step_df.groupby("model"):
        df = df[df["step"] >= MIN_STEP].sort_values("step").copy()
        df["smooth_loss"] = (
            df["step_loss"]
            .rolling(window=ROLLING_WINDOW, min_periods=1)
            .mean()
        )

        plt.plot(df["step"], df["smooth_loss"], label=model_name)

    plt.xlabel("Training step")
    plt.ylabel("Step loss, rolling mean")
    plt.title("Transformer 100-step Training Loss Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot/transformer_100step_step_loss_compare.png", dpi=300)
    plt.show()


def plot_eval_loss(eval_df):
    for metric in ["train_loss_avg", "val_loss", "srnn_loss"]:
        plt.figure(figsize=(10, 6))

        for model_name, df in eval_df.groupby("model"):
            df = df[df["step"] >= MIN_STEP].sort_values("step")
            plt.plot(df["step"], df[metric], marker="o", label=model_name)

        plt.xlabel("Training step")
        plt.ylabel(metric)
        plt.title(f"Transformer 100-step {metric} Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"plot/transformer_100step_{metric}_compare.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    step_df, eval_df = load_logs(LOG_FILES)

    step_df.to_csv("plots/transformer_100step_step_loss_parsed.csv", index=False)
    eval_df.to_csv("plots/transformer_100step_eval_loss_parsed.csv", index=False)

    plot_step_loss(step_df)
    plot_eval_loss(eval_df)

    print("\nFinal evaluation rows:")
    final_rows = (
        eval_df.sort_values("step")
        .groupby("model")
        .tail(1)
        .sort_values("val_loss")
    )
    print(final_rows.to_string(index=False))
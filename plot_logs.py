import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


MIN_PLOT_STEP = 2000

# =========================
# 1. Walking-only logs
# =========================
WALKING_LOG_FILES = {
    "RNN": "log/RNN_Seq2Seq_25steps.txt",
    "RNN + velocity": "log/RNN_residual_25steps.txt",
    "Transformer": "log/Transformer_25steps.txt",
    "Transformer + velocity": "log/Transformer_residual_25steps.txt",
}


# =========================
# 2. All-action logs
# =========================
ALL_ACTION_LOG_FILES = {
    "RNN + velocity all": "log/RNN_residual_all.txt",
    "Transformer + velocity all": "log/Transformer_residual_all.txt",
}


def parse_step_loss(text, model_name):
    pattern = re.compile(r"step\s+(\d+);\s+step_loss:\s+([0-9.]+)")
    rows = []

    for step, loss in pattern.findall(text):
        rows.append({
            "model": model_name,
            "step": int(step),
            "step_loss": float(loss)
        })

    return pd.DataFrame(rows)


def parse_eval_summary(text, model_name):
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


def parse_action_errors(text, model_name):
    table_pattern = re.compile(
        r"milliseconds\s+\|\s+80\s+\|\s+160\s+\|\s+320\s+\|\s+400\s+\|\s+560\s+\|\s+1000\s+\|\n"
        r"(.*?)\n\n============================",
        re.DOTALL
    )

    step_pattern = re.compile(r"Global step:\s+(\d+)")

    action_line = re.compile(
        r"([a-zA-Z]+)\s+\|\s+"
        r"([0-9.]+)\s+\|\s+"
        r"([0-9.]+)\s+\|\s+"
        r"([0-9.]+)\s+\|\s+"
        r"([0-9.]+)\s+\|\s+"
        r"([0-9.]+)\s+\|\s+"
        r"([0-9.]+)\s+\|"
    )

    rows = []

    tables = list(table_pattern.finditer(text))
    steps = [int(x) for x in step_pattern.findall(text)]

    for i, table_match in enumerate(tables):
        if i >= len(steps):
            continue

        step = steps[i]
        table_text = table_match.group(1)

        for match in action_line.findall(table_text):
            action, e80, e160, e320, e400, e560, e1000 = match

            rows.append({
                "model": model_name,
                "step": step,
                "action": action,
                "80ms": float(e80),
                "160ms": float(e160),
                "320ms": float(e320),
                "400ms": float(e400),
                "560ms": float(e560),
                "1000ms": float(e1000),
            })

    return pd.DataFrame(rows)


def load_logs(log_files):
    step_loss_frames = []
    eval_frames = []
    error_frames = []

    for model_name, file_path in log_files.items():
        path = Path(file_path)

        if not path.exists():
            print(f"Warning: missing log file: {path}")
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")

        step_loss_frames.append(parse_step_loss(text, model_name))
        eval_frames.append(parse_eval_summary(text, model_name))
        error_frames.append(parse_action_errors(text, model_name))

    step_loss_df = (
        pd.concat(step_loss_frames, ignore_index=True)
        if step_loss_frames else pd.DataFrame()
    )

    eval_df = (
        pd.concat(eval_frames, ignore_index=True)
        if eval_frames else pd.DataFrame()
    )

    error_df = (
        pd.concat(error_frames, ignore_index=True)
        if error_frames else pd.DataFrame()
    )

    return step_loss_df, eval_df, error_df


def plot_step_loss(step_loss_df, output_path, title):
    plt.figure(figsize=(10, 6))

    for model_name, df in step_loss_df.groupby("model"):
        df = df[df["step"] >= MIN_PLOT_STEP].sort_values("step").copy()
        df["step_loss_smooth"] = (
            df["step_loss"]
            .rolling(window=20, min_periods=1)
            .mean()
        )

        plt.plot(df["step"], df["step_loss_smooth"], label=model_name)

    plt.xlabel("Training step")
    plt.ylabel("Step loss, rolling mean")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


def plot_eval_losses(eval_df, output_prefix, title_prefix):
    for metric in ["train_loss_avg", "val_loss", "srnn_loss"]:
        plt.figure(figsize=(10, 6))

        for model_name, df in eval_df.groupby("model"):
            df = df[df["step"] >= MIN_PLOT_STEP].sort_values("step")
            plt.plot(df["step"], df[metric], marker="o", label=model_name)

        plt.xlabel("Training step")
        plt.ylabel(metric)
        plt.title(f"{title_prefix}: {metric}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_{metric}.png", dpi=300)
        plt.show()


def make_best_walking_table(error_df, eval_df, select_by="400ms"):
    rows = []

    for model_name in sorted(error_df["model"].unique()):
        err_model = error_df[
            (error_df["model"] == model_name) &
            (error_df["action"] == "walking")
        ].copy()

        if err_model.empty:
            continue

        eval_model = eval_df[eval_df["model"] == model_name].copy()

        merged = err_model.merge(
            eval_model,
            on=["model", "step"],
            how="left"
        )

        best = merged.sort_values(select_by).iloc[0]
        rows.append(best)

    result = pd.DataFrame(rows)

    keep_cols = [
        "model", "step",
        "80ms", "160ms", "320ms", "400ms", "560ms", "1000ms",
        "val_loss", "srnn_loss"
    ]

    result = result[keep_cols]
    result = result.sort_values("400ms")

    return result


def make_all_action_mean_table(error_df, eval_df, select_step=None):
    rows = []

    for model_name, df in error_df.groupby("model"):
        if select_step is None:
            use_step = df["step"].max()
        else:
            use_step = select_step

        df_step = df[df["step"] == use_step]

        if df_step.empty:
            continue

        eval_model = eval_df[
            (eval_df["model"] == model_name) &
            (eval_df["step"] == use_step)
        ]

        if not eval_model.empty:
            val_loss = eval_model.iloc[0]["val_loss"]
            srnn_loss = eval_model.iloc[0]["srnn_loss"]
        else:
            val_loss = None
            srnn_loss = None

        rows.append({
            "model": model_name,
            "step": use_step,
            "mean_80ms": df_step["80ms"].mean(),
            "mean_160ms": df_step["160ms"].mean(),
            "mean_320ms": df_step["320ms"].mean(),
            "mean_400ms": df_step["400ms"].mean(),
            "mean_560ms": df_step["560ms"].mean(),
            "mean_1000ms": df_step["1000ms"].mean(),
            "val_loss": val_loss,
            "srnn_loss": srnn_loss,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("mean_400ms")

    return result


def make_all_action_per_action_table(error_df, step=None):
    rows = []

    for model_name, df in error_df.groupby("model"):
        if step is None:
            use_step = df["step"].max()
        else:
            use_step = step

        df_step = df[df["step"] == use_step].copy()

        if df_step.empty:
            continue

        df_step = df_step[
            ["model", "step", "action", "80ms", "160ms", "320ms", "400ms", "560ms", "1000ms"]
        ]

        rows.append(df_step)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(["action", "model"])

    return result


def process_walking_only():
    print("\n==============================")
    print("Processing walking-only logs")
    print("==============================")

    step_loss_df, eval_df, error_df = load_logs(WALKING_LOG_FILES)

    step_loss_df.to_csv("walking_step_loss.csv", index=False)
    eval_df.to_csv("walking_eval_losses.csv", index=False)
    error_df.to_csv("walking_action_errors.csv", index=False)

    plot_step_loss(
        step_loss_df,
        output_path="walking_step_loss_curve.png",
        title="Walking-only Training Step Loss"
    )

    plot_eval_losses(
        eval_df,
        output_prefix="walking_eval_loss",
        title_prefix="Walking-only"
    )

    walking_best = make_best_walking_table(
        error_df,
        eval_df,
        select_by="400ms"
    )

    walking_best.to_csv("walking_best_checkpoint_table.csv", index=False)

    print("\nWalking-only best checkpoint table:")
    print(walking_best.to_string(index=False))


def process_all_action():
    print("\n==============================")
    print("Processing all-action logs")
    print("==============================")

    step_loss_df, eval_df, error_df = load_logs(ALL_ACTION_LOG_FILES)

    step_loss_df.to_csv("all_action_step_loss.csv", index=False)
    eval_df.to_csv("all_action_eval_losses.csv", index=False)
    error_df.to_csv("all_action_errors.csv", index=False)

    plot_step_loss(
        step_loss_df,
        output_path="all_action_step_loss_curve.png",
        title="All-action Training Step Loss"
    )

    plot_eval_losses(
        eval_df,
        output_prefix="all_action_eval_loss",
        title_prefix="All-action"
    )

    all_action_mean = make_all_action_mean_table(
        error_df,
        eval_df,
        select_step=None
    )

    all_action_per_action = make_all_action_per_action_table(
        error_df,
        step=None
    )

    all_action_mean.to_csv("all_action_mean_error_table.csv", index=False)
    all_action_per_action.to_csv("all_action_per_action_error_table.csv", index=False)

    print("\nAll-action mean error table:")
    print(all_action_mean.to_string(index=False))

    print("\nAll-action per-action error table:")
    print(all_action_per_action.to_string(index=False))


if __name__ == "__main__":
    process_walking_only()
    process_all_action()
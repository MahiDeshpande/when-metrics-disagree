#!/usrusr/bin/env python3
"""Create standard and radar TPIPS prompt sensitivity visualizations."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "tpips_prompt_experiments_master.csv"
OUTPUT_DIR = ROOT / "evaluation_figures"

SEVERITY_ORDER = {
    "Gaussian Noise": [5, 10, 20, 40],
    "Gaussian Blur": [1, 2, 4, 8],
    "JPEG Compression": [90, 70, 40, 10],
    "Spatial Translation": [1, 2, 4, 8],
    "Brightness Shift": [0.7, 0.9, 1.1, 1.3],
    "Contrast Alterations": [0.7, 1.3],
    "Saturation Alterations": [0, 0.5, 1.5],
    "Spatial Rotation": [1, 3, 5, 10],
    "Crop + Resize": [2, 5, 10, 20],
    "Hue Shift": [10, 30, 60, 90],
}

PARAMETER_LABELS = {
    "Gaussian Noise": lambda v: f"σ {int(v)}",
    "Gaussian Blur": lambda v: f"r {int(v)} px",
    "JPEG Compression": lambda v: f"Q {int(v)}",
    "Spatial Translation": lambda v: f"{int(v)} px",
    "Brightness Shift": lambda v: f"×{v:g}",
    "Contrast Alterations": lambda v: f"×{v:g}",
    "Saturation Alterations": lambda v: f"×{v:g}",
    "Spatial Rotation": lambda v: f"{int(v)}°",
    "Crop + Resize": lambda v: f"{v:g}%",
    "Hue Shift": lambda v: f"{int(v)}°",
}


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text.lower()).strip("_")


def prompt_title(column: str) -> str:
    return column.removeprefix("Prompt_").replace("_", " ").title()


def load_data() -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(DATA_CSV)
    prompt_columns = [column for column in frame.columns if column.startswith("Prompt_")]
    frame[prompt_columns] = frame[prompt_columns].apply(pd.to_numeric, errors="coerce")
    return frame, prompt_columns


def test_prompts(frame: pd.DataFrame, prompt_columns: list[str], distortion: str) -> list[str]:
    test_frame = frame[frame["Distortion Type"] == distortion]
    return [
        column for column in prompt_columns
        if test_frame[column].notna().any()
    ]


def make_line_chart(frame: pd.DataFrame, distortion: str, prompts: list[str]) -> Path:
    test_frame = frame[frame["Distortion Type"] == distortion]
    mean_scores = (
        test_frame.groupby("Parameter Value")[prompts]
        .mean()
        .reset_index()
    )
    values = SEVERITY_ORDER.get(distortion, sorted(mean_scores["Parameter Value"].unique()))
    mean_scores = mean_scores.set_index("Parameter Value").reindex(values).reset_index()

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 7))
    palette = sns.color_palette("tab10", n_colors=len(prompts))

    for prompt, color in zip(prompts, palette):
        ax.plot(
            values,
            mean_scores[prompt],
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=prompt_title(prompt),
            color=color,
        )

    ax.set_xscale("log" if distortion == "JPEG Compression" else "linear")
    ax.set_xticks(values)
    labels = PARAMETER_LABELS.get(distortion, lambda value: str(value))
    ax.set_xticklabels([labels(value) for value in values])
    ax.set_xlabel("Distortion Severity Level")
    ax.set_ylabel("Mean TPIPS Distance")
    ax.set_title(f"{distortion}: TPIPS Prompt Sensitivity")
    ax.legend(title="TPIPS Prompt", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    path = OUTPUT_DIR / f"tpips_{safe_name(distortion)}_prompt_lines.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def make_radar_chart(frame: pd.DataFrame, distortion: str, prompts: list[str]) -> Path:
    test_frame = frame[frame["Distortion Type"] == distortion]
    severity = SEVERITY_ORDER.get(distortion, sorted(test_frame["Parameter Value"].unique()))
    scores = test_frame.groupby("Parameter Value")[prompts].mean().reindex(severity)

    angles = np.linspace(0, 2 * np.pi, len(prompts), endpoint=False).tolist()
    angles += angles[:1]
    colors = sns.color_palette("Reds", n_colors=len(severity))

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([prompt_title(prompt) for prompt in prompts], fontsize=10)

    severity_labels = PARAMETER_LABELS.get(distortion, lambda value: str(value))
    for index, value in enumerate(severity):
        polygon = scores.loc[value].tolist()
        polygon += polygon[:1]
        ax.plot(angles, polygon, marker="o", linewidth=2, label=f"Level {index + 1}: {severity_labels(value)}", color=colors[index])
        ax.fill(angles, polygon, alpha=0.08, color=colors[index])

    ax.set_title(f"{distortion}: Prompt Sensitivity by Severity", pad=34)
    ax.set_ylabel("Mean TPIPS Distance", labelpad=38)
    ax.legend(title="Severity Level", loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=min(2, len(severity)))
    fig.tight_layout()

    path = OUTPUT_DIR / f"tpips_{safe_name(distortion)}_prompt_radar.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    frame, prompt_columns = load_data()
    generated = []

    distortions = list(frame["Distortion Type"].drop_duplicates())
    for distortion in distortions:
        prompts = test_prompts(frame, prompt_columns, distortion)
        if not prompts:
            continue
        generated.append(make_line_chart(frame, distortion, prompts))
        generated.append(make_radar_chart(frame, distortion, prompts))

    print(f"Processed distortions: {len(distortions)}")
    print(f"Generated figures: {len(generated)}")
    for path in generated:
        print(path.name)


if __name__ == "__main__":
    main()

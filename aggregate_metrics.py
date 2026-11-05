#!/usr/bin/env python3
"""Aggregate experiment metrics from repository data and notebook sources."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from io import StringIO


ROOT = Path(__file__).resolve().parent
DATA_JSON = ROOT / "docs" / "data.json"
TPIPS_JS = ROOT / "docs" / "tpips-data.js"
OUTPUT_CSV = ROOT / "master_evaluation_dataset.csv"
FIGURE_DIR = ROOT / "evaluation_figures"

DISTORTION_ALIASES = {
    "gaussian_noise": "Gaussian Noise",
    "gaussian_blur": "Gaussian Blur",
    "jpeg": "JPEG Compression",
    "translation": "Spatial Translation",
    "brightness": "Brightness Shift",
    "contrast": "Contrast Alterations",
    "saturation": "Saturation Alterations",
    "rotation": "Spatial Rotation",
    "crop_resize": "Crop + Resize",
    "hue": "Hue Shift",
}

PQI_ANCHORS = {
    "PSNR": [(10, 0), (20, 25), (30, 50), (40, 75), (50, 100)],
    "SSIM": [(0, 0), (0.6, 25), (0.85, 50), (0.95, 75), (1, 100)],
    "LPIPS-Alex": [(0, 100), (0.1, 75), (0.3, 50), (0.6, 25), (1, 0)],
    "LPIPS-VGG": [(0, 100), (0.1, 75), (0.3, 50), (0.6, 25), (1, 0)],
    "TPIPS Distance": [(0, 100), (0.05, 75), (0.15, 50), (0.3, 25), (0.6, 0)],
}


def interpolate_pqi(metric: str, value: float) -> float:
    points = PQI_ANCHORS[metric]
    ascending = points[0][0] < points[-1][0]
    if not ascending:
        points = list(reversed(points))
    if value <= points[0][0]:
        return float(points[0][1])
    if value >= points[-1][0]:
        return float(points[-1][1])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= value <= x1:
            return y0 + ((value - x0) / (x1 - x0)) * (y1 - y0)
    return 0.0


def notebook_image_ids(root: Path) -> list[str]:
    """Scan notebook names and their JSON metadata for image identifiers."""
    image_ids = set()
    for notebook in sorted(root.rglob("*.ipynb")):
        try:
            payload = json.loads(notebook.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        notebook_text = json.dumps(payload, ensure_ascii=False)
        for match in re.findall(r"ILSVRC2012_val_\d{8}", notebook_text + notebook.name):
            image_ids.add(match)
    return sorted(image_ids)


def parse_exported_tpips(path: Path) -> dict[tuple[str, str], float]:
    text = path.read_text(encoding="utf-8")
    assignment = re.search(r"=\s*(\[\{.*\}\]);\s*$", text, re.DOTALL)
    if not assignment:
        return {}
    tables = json.loads(assignment.group(1))
    values = {}
    for table in tables:
        headers = table["headers"]
        try:
            overall_index = next(
                i for i, header in enumerate(headers) if "TPIPS (overall)" in header
            )
        except StopIteration:
            continue
        for row in table["rows"]:
            values[(table["name"], str(row[0]))] = float(row[overall_index - 1])
    return values


def parse_notebook_tpips(root: Path) -> dict[tuple[str, str, float], float]:
    values = {}
    for notebook in sorted(root.glob("TPIPS_EXPERIMENTATION*.ipynb")):
        image_match = re.search(r"(ILSVRC2012_val_\d{8})", notebook.name)
        if not image_match:
            continue
        image_id = image_match.group(1)
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        for cell in payload.get("cells", []):
            source = "".join(cell.get("source", []))
            title_match = re.search(r'exp_title="([^"]+)"', source)
            if not title_match:
                continue
            title = title_match.group(1)
            canonical_title = {
                "Gaussian Noise": "Gaussian Noise",
                "Gaussian Noise Sensitivity": "Gaussian Noise",
                "Gaussian Blur": "Gaussian Blur",
                "JPEG Compression": "JPEG Compression",
                "Spatial Translation": "Spatial Translation",
                "Brightness Shift": "Brightness Shift",
                "Contrast Alterations": "Contrast Alterations",
                "Saturation Alterations": "Saturation Alterations",
                "Spatial Rotation": "Spatial Rotation",
                "Crop + Resize": "Crop + Resize",
                "Color Shift (Hue)": "Hue Shift",
                "Color Shift": "Hue Shift",
                "Hue Shift": "Hue Shift",
            }.get(title)
            if canonical_title is None:
                continue
            for output in cell.get("outputs", []):
                html_parts = output.get("data", {}).get("text/html", [])
                if not html_parts:
                    continue
                html = "".join(html_parts)
                if "TPIPS (overall)" not in html:
                    continue
                tables = pd.read_html(StringIO(html))
                for table in tables:
                    overall_column = next(
                        column for column in table.columns
                        if isinstance(column, str) and "TPIPS (overall)" in column
                    )
                    parameter_column = next(
                        column for column in table.columns
                        if isinstance(column, str) and column != "Unnamed: 0"
                        and "TPIPS" not in column
                    )
                    for _, row in table.iterrows():
                        try:
                            parameter = float(row[parameter_column])
                            distance = float(row[overall_column])
                        except (TypeError, ValueError):
                            continue
                        values[(image_id, canonical_title, parameter)] = distance
    return values


def parse_sketch_metrics(path: Path) -> dict[tuple[str, float], dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = {}
    titles = {
        "Gaussian Noise": "Gaussian Noise",
        "Gaussian Noise Sensitivity": "Gaussian Noise",
        "Gaussian Blur": "Gaussian Blur",
        "JPEG Compression": "JPEG Compression",
        "Spatial Translation": "Spatial Translation",
        "Brightness Shift": "Brightness Shift",
        "Contrast Alterations": "Contrast Alterations",
        "Saturation Alterations": "Saturation Alterations",
        "Spatial Rotation": "Spatial Rotation",
        "Crop + Resize": "Crop + Resize",
        "Color Shift (Hue)": "Hue Shift",
        "Color Shift": "Hue Shift",
        "Hue Shift": "Hue Shift",
    }
    metric_columns = {
        "PSNR": "PSNR",
        "SSIM": "SSIM",
        "LPIPS-Alex": "LPIPS-Alex",
        "LPIPS-VGG": "LPIPS-VGG",
        "TPIPS Distance": "TPIPS Distance",
    }

    for cell in payload.get("cells", []):
        source = "".join(cell.get("source", []))
        title_match = re.search(r'exp_title="([^"]+)"', source)
        if not title_match and "TEST 1" in source:
            title_match = re.match(r".*TEST 1:.*", source, re.DOTALL)
        if title_match:
            title_text = title_match.group(1)
        else:
            continue
        distortion = titles.get(title_text)
        if distortion is None:
            continue

        for output in cell.get("outputs", []):
            html_parts = output.get("data", {}).get("text/html", [])
            if not html_parts:
                continue
            html = "".join(html_parts)
            if "PSNR" not in html or "TPIPS Distance" not in html:
                continue
            for table in pd.read_html(StringIO(html)):
                parameter_column = next(
                    column for column in table.columns
                    if isinstance(column, str)
                    and "PSNR" not in column
                    and "SSIM" not in column
                    and "LPIPS" not in column
                    and "TPIPS" not in column
                    and column != "Unnamed: 0"
                )
                for _, row in table.iterrows():
                    try:
                        parameter = float(row[parameter_column])
                    except (TypeError, ValueError):
                        continue
                    record = {}
                    for output_name, column in metric_columns.items():
                        matching = next(
                            (
                                table_column for table_column in table.columns
                                if isinstance(table_column, str)
                                and output_name.split()[0] in table_column
                            ),
                            None,
                        )
                        if matching is not None:
                            try:
                                record[column] = float(row[matching])
                            except (TypeError, ValueError):
                                pass
                    if record:
                        values[(distortion, parameter)] = record
    return values


def format_parameter(distortion_id: str, value: float) -> str:
    if float(value).is_integer():
        rendered = str(int(value))
    else:
        rendered = f"{float(value):g}"
    suffix = {
        "gaussian_noise": f"Sigma {rendered}",
        "gaussian_blur": f"Radius {rendered}",
        "jpeg": f"Quality {rendered}",
        "translation": f"Shift {rendered}px",
        "brightness": f"Factor {rendered}",
        "contrast": f"Factor {rendered}",
        "saturation": f"Factor {rendered}",
        "rotation": f"Degrees {rendered}",
        "crop_resize": f"Crop {rendered}%",
        "hue": f"Shift {rendered}\u00b0",
    }
    return suffix[distortion_id]


def severity_number(distortion_id: str, value: float) -> float:
    if distortion_id == "jpeg":
        return -float(value)
    return float(value)


def load_master() -> pd.DataFrame:
    source = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    notebooks = sorted(ROOT.rglob("*.ipynb"))
    image_ids = {image["id"] for image in source["images"]}
    image_ids.update(notebook_image_ids(ROOT))

    rows = []
    tpips = parse_notebook_tpips(ROOT)
    sketch_metrics = parse_sketch_metrics(ROOT / "when_metrics_disagree_sketch.ipynb")
    for distortion_id, levels_by_image in source["results"].items():
        distortion = next(
            item for item in source["distortions"] if item["id"] == distortion_id
        )
        for image_id in sorted(set(levels_by_image) & image_ids):
            for level_index, level in enumerate(levels_by_image[image_id]):
                parameter_value = float(distortion["levels"][level_index])
                row = {
                    "Test Name": DISTORTION_ALIASES[distortion_id],
                    "Parameter": format_parameter(distortion_id, parameter_value),
                    "Parameter Value": parameter_value,
                    "Severity Value": severity_number(distortion_id, parameter_value),
                    "Image": image_id,
                    "PSNR": level["psnr"],
                    "SSIM": level["ssim"],
                    "LPIPS-Alex": level["lpips_alex"],
                    "LPIPS-VGG": level["lpips_vgg"],
                }
                if image_id == "img_ref":
                    row.update(sketch_metrics.get((DISTORTION_ALIASES[distortion_id], parameter_value), {}))
                tpips_value = tpips.get((image_id, DISTORTION_ALIASES[distortion_id], parameter_value))
                if tpips_value is None:
                    exported_tpips = parse_exported_tpips(TPIPS_JS)
                    tpips_value = exported_tpips.get((DISTORTION_ALIASES[distortion_id], str(parameter_value)))
                    if tpips_value is None:
                        tpips_value = exported_tpips.get((DISTORTION_ALIASES[distortion_id], str(int(parameter_value))))
                if tpips_value is not None:
                    row["TPIPS Distance"] = tpips_value
                rows.append(row)

    frame = pd.DataFrame(rows)
    frame["TPIPS Distance"] = pd.to_numeric(frame["TPIPS Distance"], errors="coerce") if "TPIPS Distance" in frame else float("nan")
    numeric_metrics = list(PQI_ANCHORS)
    for metric in numeric_metrics:
        if metric in frame:
            frame[f"{metric} Rating"] = frame[metric].map(
                lambda value, metric=metric: interpolate_pqi(metric, value)
                if pd.notna(value) else float("nan")
            )
    rating_columns = [f"{metric} Rating" for metric in numeric_metrics if f"{metric} Rating" in frame]
    frame["PQI"] = frame[rating_columns].mean(axis=1, skipna=True)
    frame["PQI Tier"] = pd.cut(
        frame["PQI"],
        bins=[-0.001, 25, 50, 75, 100],
        labels=["Heavy", "Noticeable", "Slight", "Indistinguishable"],
        include_lowest=True,
    ).astype(str)
    frame["Source Notebook Count"] = len(notebooks)
    return frame.sort_values(["Test Name", "Severity Value", "Image"]).reset_index(drop=True)


def generate_figures(frame: pd.DataFrame) -> list[Path]:
    FIGURE_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    generated = []

    metrics = ["PSNR", "SSIM", "LPIPS-Alex", "LPIPS-VGG", "TPIPS Distance"]
    palette = sns.color_palette("tab20", n_colors=frame["Image"].nunique())
    image_colors = dict(zip(sorted(frame["Image"].unique()), palette))

    for test_name, test_frame in frame.groupby("Test Name", sort=True):
        available_metrics = [
            metric for metric in metrics
            if metric in test_frame and test_frame[metric].notna().any()
        ]
        fig, axes = plt.subplots(
            1,
            len(available_metrics),
            figsize=(5.2 * len(available_metrics), 6),
            squeeze=False,
        )
        severity_orders = {
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
        test_parameters = test_frame[
            ["Parameter", "Parameter Value"]
        ].drop_duplicates()
        parameter_labels = dict(zip(test_parameters["Parameter Value"], test_parameters["Parameter"]))
        parameter_order = severity_orders.get(
            test_name,
            sorted(test_parameters["Parameter Value"].unique()),
        )

        for ax, metric in zip(axes[0], available_metrics):
            sns.lineplot(
                data=test_frame,
                x="Parameter",
                y=metric,
                hue="Image",
                palette=image_colors,
                marker="o",
                linewidth=1.8,
                ax=ax,
                legend=False,
                sort=False,
            )
            ax.set_title(metric)
            ax.set_xlabel("Severity")
            ax.set_ylabel(metric)
            ax.set_xticks(range(len(parameter_order)))
            ax.set_xticklabels(
                [parameter_labels[value] for value in parameter_order],
                rotation=45,
                ha="right",
            )

        handles = [
            plt.Line2D([0], [0], color=image_colors[image], marker="o", linestyle="-", label=image)
            for image in sorted(test_frame["Image"].unique())
        ]
        fig.legend(
            handles=handles,
            title="Image",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
        fig.suptitle(f"{test_name}: Metric Degradation by Severity", y=1.02)
        fig.tight_layout(rect=(0, 0, 0.85, 1))
        stem = re.sub(r"[^A-Za-z0-9]+", "_", test_name.lower()).strip("_")
        png = FIGURE_DIR / f"{stem}_degradation_by_severity.png"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        generated.append(png)

    for objective in ["PSNR", "LPIPS-Alex", "SSIM", "TPIPS Distance"]:
        if objective not in frame or frame[objective].notna().sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=frame.dropna(subset=[objective, "PQI"]),
            x=objective,
            y="PQI",
            hue="Test Name",
            alpha=0.75,
            ax=ax,
        )
        sns.regplot(
            data=frame.dropna(subset=[objective, "PQI"]),
            x=objective,
            y="PQI",
            scatter=False,
            color="black",
            ci=95,
            ax=ax,
        )
        ax.set_xlabel(objective)
        ax.set_ylabel("Perceptual Quality Index (PQI)")
        ax.set_title(f"PQI vs. {objective}")
        ax.legend(title="Distortion", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        stem = objective.lower().replace("-", "_").replace(" ", "_")
        png = FIGURE_DIR / f"pqi_vs_{stem}.png"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        generated.append(png)
    return generated


def main() -> None:
    frame = load_master()
    frame.to_csv(OUTPUT_CSV, index=False)
    generated = generate_figures(frame)
    print(
        json.dumps(
            {
                "shape": list(frame.shape),
                "rows": len(frame),
                "columns": list(frame.columns),
                "notebooks_scanned": int(frame["Source Notebook Count"].iloc[0]),
                "csv": str(OUTPUT_CSV),
                "figures": [str(path) for path in generated],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

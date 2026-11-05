#!/usr/bin/env python3
"""Extract TPIPS prompt scores from the TPIPS experiment notebooks."""

import json
import re
from io import StringIO
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_CSV = ROOT / "tpips_prompt_experiments_master.csv"


def prompt_column(prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    return f"Prompt_{slug}"


def parse_html_table(html: str) -> pd.DataFrame | None:
    if "TPIPS (" not in html:
        return None
    try:
        tables = pd.read_html(StringIO(html))
    except (ValueError, ImportError):
        return None
    return tables[0] if tables else None


def extract_prompt(prompt_cell_name: str) -> str | None:
    match = re.search(r"TPIPS\s*\(([^)]+)\)", prompt_cell_name)
    return match.group(1) if match else None


def canonical_distortion(source: str) -> str | None:
    match = re.search(r'exp_title="([^"]+)"', source)
    if not match:
        return None
    title = match.group(1)
    aliases = {
        "Color Shift (Hue)": "Hue Shift",
        "Color Shift": "Hue Shift",
    }
    return aliases.get(title, title)


def extract_tpips_prompt_data() -> pd.DataFrame:
    rows = []
    prompt_columns = set()

    for notebook in sorted(ROOT.glob("TPIPS_EXPERIMENTATION*.ipynb")):
        image_match = re.search(r"(ILSVRC2012_val_\d{8})", notebook.name)
        if not image_match:
            continue
        image_id = image_match.group(1)
        payload = json.loads(notebook.read_text(encoding="utf-8"))

        for cell in payload.get("cells", []):
            source = "".join(cell.get("source", []))
            distortion = canonical_distortion(source)
            if distortion is None:
                continue

            for output in cell.get("outputs", []):
                html_parts = output.get("data", {}).get("text/html", [])
                if not html_parts:
                    continue
                table = parse_html_table("".join(html_parts))
                if table is None:
                    continue

                parameter_columns = [
                    column for column in table.columns
                    if isinstance(column, str)
                    and "TPIPS (" not in column
                    and column != "Unnamed: 0"
                ]
                if not parameter_columns:
                    continue
                parameter_column = parameter_columns[0]

                score_columns = [
                    column for column in table.columns
                    if isinstance(column, str) and "TPIPS (" in column
                ]
                if not score_columns:
                    continue

                for _, result in table.iterrows():
                    try:
                        parameter = float(result[parameter_column])
                    except (TypeError, ValueError):
                        continue
                    row = {
                        "Image ID": image_id,
                        "Distortion Type": distortion,
                        "Parameter Value": parameter,
                    }
                    for score_column in score_columns:
                        prompt = extract_prompt(score_column)
                        if prompt is None:
                            continue
                        column = prompt_column(prompt)
                        try:
                            row[column] = float(result[score_column])
                        except (TypeError, ValueError):
                            continue
                        prompt_columns.add(column)
                    rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No TPIPS prompt tables were found in the notebooks.")

    key_columns = ["Image ID", "Distortion Type", "Parameter Value"]
    ordered_prompt_columns = sorted(prompt_columns)
    frame = frame[key_columns + ordered_prompt_columns].drop_duplicates(
        subset=key_columns,
        keep="last",
    )
    return frame.sort_values(key_columns).reset_index(drop=True)


def main() -> None:
    frame = extract_tpips_prompt_data()
    frame.to_csv(OUTPUT_CSV, index=False)
    prompt_columns = [
        column for column in frame.columns if column.startswith("Prompt_")
    ]
    print(f"Shape: {frame.shape}")
    print("Prompt columns:")
    for column in prompt_columns:
        print(f"- {column}")
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

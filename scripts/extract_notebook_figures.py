"""Extract report-relevant figures from executed Jupyter notebooks.

The script reads stored notebook outputs only. It does not execute notebooks.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
}

REPORT_KEYWORDS = {
    "action",
    "actor",
    "alpha",
    "baseline",
    "capacity",
    "comparison",
    "critic",
    "delay",
    "discount",
    "double",
    "dqn",
    "dueling",
    "emergency",
    "episode",
    "epsilon",
    "evaluation",
    "gamma",
    "grid",
    "heatmap",
    "icu",
    "learning",
    "ppo",
    "q-learning",
    "qtable",
    "q-table",
    "random",
    "rejection",
    "reject",
    "reward",
    "training",
    "utilisation",
    "utilization",
}

PLOT_TOKENS = (
    "plt.",
    ".plot(",
    ".bar(",
    "sns.",
    "imshow",
    "heatmap",
    "scatter(",
    "hist(",
    "Figure(",
    "plot_metric",
    "plot_results",
)

IRRELEVANT_KEYWORDS = {
    "install",
    "pip",
    "requirements",
    "warning",
    "traceback",
    "error",
    "download",
    "logo",
    "screenshot",
}


@dataclass
class ExtractedFigure:
    filename: str
    source_notebook: str
    cell_index: int
    output_type: str
    guessed_topic: str
    nearby_heading: str
    report_relevant: bool
    reason_kept: str


@dataclass
class MissingPlot:
    notebook: str
    cell_index: int
    reason: str
    likely_topic: str
    nearby_heading: str


def normalise_slug(value: str, max_len: int = 70) -> str:
    value = value.lower().replace("+", " plus ")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    value = re.sub(r"_+", "_", value)
    return (value[:max_len].strip("_") or "figure")


def flatten_text(value) -> str:
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    return str(value or "")


def image_data_to_bytes(value) -> bytes:
    raw = flatten_text(value).strip()
    if raw.startswith("data:image"):
        raw = raw.split(",", 1)[-1]
    return base64.b64decode(raw)


def iter_notebooks(root: Path, include_checkpoints: bool) -> Iterable[Path]:
    for path in sorted(root.rglob("*.ipynb")):
        parts = {part.lower() for part in path.parts}
        if not include_checkpoints and ".ipynb_checkpoints" in parts:
            continue
        yield path


def markdown_heading_from_source(source: str) -> str:
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def nearest_heading(cells: list[dict], cell_index: int) -> str:
    for idx in range(cell_index, -1, -1):
        cell = cells[idx]
        if cell.get("cell_type") == "markdown":
            heading = markdown_heading_from_source(flatten_text(cell.get("source", "")))
            if heading:
                return heading
    return ""


def extract_plot_strings(source: str) -> list[str]:
    patterns = [
        r"plt\\.title\\(([^)]*)\\)",
        r"\\.set_title\\(([^)]*)\\)",
        r"plt\\.xlabel\\(([^)]*)\\)",
        r"plt\\.ylabel\\(([^)]*)\\)",
        r"label\\s*=\\s*([\"'][^\"']+[\"'])",
    ]
    found: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, source):
            found.append(match.group(1).strip("\"' "))
    return found


def code_comment_summary(source: str) -> str:
    comments = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("# %%"):
            comments.append(stripped.lstrip("#").strip())
        if len(comments) >= 3:
            break
    return " | ".join(comments)


def guess_topic(notebook_name: str, heading: str, source: str) -> str:
    combined = " ".join(
        [notebook_name, heading, code_comment_summary(source), " ".join(extract_plot_strings(source))]
    ).lower()
    topic_rules = [
        ("ppo_vs_random_baseline", ("ppo", "random")),
        ("ppo_training_reward_curve", ("ppo", "reward")),
        ("dqn_comparison", ("dqn", "comparison")),
        ("dueling_double_dqn", ("dueling", "dqn")),
        ("emergency_rejections", ("emergency", "rejection")),
        ("icu_utilisation", ("icu", "utilisation")),
        ("icu_utilisation", ("icu", "utilization")),
        ("q_learning_heatmap", ("q-learning", "heatmap")),
        ("q_learning_heatmap", ("q table", "heatmap")),
        ("hyperparameter_sweep", ("learning", "rate")),
        ("hyperparameter_sweep", ("discount", "factor")),
        ("action_frequency", ("action", "count")),
        ("grid_search", ("grid", "search")),
        ("episode_reward", ("episode", "reward")),
        ("actor_critic", ("actor", "critic")),
        ("capacity_increase", ("capacity",)),
    ]
    for topic, terms in topic_rules:
        if all(term in combined for term in terms):
            return topic

    matches = [keyword for keyword in sorted(REPORT_KEYWORDS) if keyword in combined]
    if matches:
        return normalise_slug("_".join(matches[:3]))
    return "notebook_plot"


def relevance_reason(notebook_name: str, heading: str, source: str) -> tuple[bool, str]:
    combined = " ".join(
        [notebook_name, heading, code_comment_summary(source), " ".join(extract_plot_strings(source)), source]
    ).lower()
    has_plot_code = any(token.lower() in combined for token in PLOT_TOKENS)
    matches = [keyword for keyword in sorted(REPORT_KEYWORDS) if keyword in combined]
    irrelevant_matches = [keyword for keyword in sorted(IRRELEVANT_KEYWORDS) if keyword in combined]

    if matches:
        return True, "matches report keywords: " + ", ".join(matches[:8])
    if has_plot_code and not irrelevant_matches:
        return True, "plotting code output with no irrelevant debug/install keywords"
    return False, "no report-analysis keyword or relevant plotting context detected"


def unique_output_path(out_dir: Path, filename: str, overwrite: bool) -> Path | None:
    candidate = out_dir / filename
    if overwrite or not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    for idx in range(1, 1000):
        alt = out_dir / f"{stem}_{idx}{suffix}"
        if not alt.exists():
            return alt
    return None


def write_index(out_dir: Path, rows: list[ExtractedFigure]) -> None:
    index_path = out_dir / "figure_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "source_notebook",
                "cell_index",
                "output_type",
                "guessed_topic",
                "nearby_markdown_heading_or_comments",
                "report_relevant",
                "reason_kept",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "filename": row.filename,
                    "source_notebook": row.source_notebook,
                    "cell_index": row.cell_index,
                    "output_type": row.output_type,
                    "guessed_topic": row.guessed_topic,
                    "nearby_markdown_heading_or_comments": row.nearby_heading,
                    "report_relevant": row.report_relevant,
                    "reason_kept": row.reason_kept,
                }
            )


def write_readme(out_dir: Path, rows: list[ExtractedFigure]) -> None:
    grouped: dict[str, list[ExtractedFigure]] = {}
    for row in rows:
        grouped.setdefault(row.source_notebook, []).append(row)

    lines = [
        "# Extracted Notebook Figures",
        "",
        "These figures were extracted from stored Jupyter notebook outputs.",
        "The notebooks were not rerun or modified.",
        "",
    ]
    for notebook, notebook_rows in sorted(grouped.items()):
        lines.append(f"## {notebook}")
        lines.append("")
        for row in notebook_rows:
            lines.append(
                f"- `{row.filename}`: {row.guessed_topic.replace('_', ' ')} "
                f"(cell {row.cell_index}). {row.reason_kept}."
            )
        lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_missing_plots(out_dir: Path, rows: list[MissingPlot]) -> None:
    lines = [
        "# Missing Or Unsaved Notebook Plots",
        "",
        "These cells contain plotting code but no stored image output was found.",
        "They may need rerunning if the figure is required for the report.",
        "",
    ]
    if not rows:
        lines.append("No missing or unsaved plotting cells were detected.")
    else:
        for row in rows:
            lines.extend(
                [
                    f"## {row.notebook} cell {row.cell_index}",
                    f"- Likely topic: {row.likely_topic}",
                    f"- Nearby heading/comments: {row.nearby_heading or 'not detected'}",
                    f"- Reason: {row.reason}",
                    "",
                ]
            )
    (out_dir / "missing_or_unsaved_plots.md").write_text("\n".join(lines), encoding="utf-8")


def extract_figures(root: Path, out_dir: Path, overwrite: bool, include_checkpoints: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted_rows: list[ExtractedFigure] = []
    missing_rows: list[MissingPlot] = []
    notebooks_scanned = 0
    image_outputs_found = 0
    image_outputs_relevant = 0

    for notebook_path in iter_notebooks(root, include_checkpoints):
        notebooks_scanned += 1
        rel_notebook = notebook_path.relative_to(root).as_posix()
        notebook_stem = normalise_slug(notebook_path.stem, max_len=45)
        try:
            nb = json.loads(notebook_path.read_text(encoding="utf-8"))
        except Exception as exc:
            missing_rows.append(
                MissingPlot(rel_notebook, -1, f"notebook could not be parsed: {exc}", "unknown", "")
            )
            continue

        cells = nb.get("cells", [])
        for cell_index, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue

            source = flatten_text(cell.get("source", ""))
            heading = nearest_heading(cells, cell_index)
            comments = code_comment_summary(source)
            context = heading or comments
            topic = guess_topic(notebook_path.name, heading, source)
            report_relevant, reason = relevance_reason(notebook_path.name, heading, source)

            cell_image_count = 0
            for output_index, output in enumerate(cell.get("outputs", [])):
                data = output.get("data", {}) if isinstance(output, dict) else {}
                for mime, ext in IMAGE_MIME_TO_EXT.items():
                    if mime not in data:
                        continue
                    image_outputs_found += 1
                    if not report_relevant:
                        continue

                    image_outputs_relevant += 1
                    filename = (
                        f"{notebook_stem}_cell_{cell_index:03d}_"
                        f"output_{output_index:02d}_{topic}{ext}"
                    )
                    output_path = unique_output_path(out_dir, filename, overwrite)
                    if output_path is None:
                        continue
                    output_path.write_bytes(image_data_to_bytes(data[mime]))

                    extracted_rows.append(
                        ExtractedFigure(
                            filename=output_path.name,
                            source_notebook=rel_notebook,
                            cell_index=cell_index,
                            output_type=mime,
                            guessed_topic=topic,
                            nearby_heading=context,
                            report_relevant=True,
                            reason_kept=reason,
                        )
                    )
                    cell_image_count += 1

            if cell_image_count == 0 and any(token in source for token in PLOT_TOKENS):
                missing_rows.append(
                    MissingPlot(
                        notebook=rel_notebook,
                        cell_index=cell_index,
                        reason="plotting code found but no stored image output was available",
                        likely_topic=topic,
                        nearby_heading=context,
                    )
                )

    write_index(out_dir, extracted_rows)
    write_readme(out_dir, extracted_rows)
    write_missing_plots(out_dir, missing_rows)

    return {
        "notebooks_scanned": notebooks_scanned,
        "image_outputs_found": image_outputs_found,
        "report_relevant_image_outputs": image_outputs_relevant,
        "figures_extracted": len(extracted_rows),
        "missing_or_unsaved_plot_cells": len(missing_rows),
        "output_dir": str(out_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument(
        "--out",
        default="report_assets/figures/from_notebooks",
        help="Output folder for extracted figures and index files.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing figure files.")
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Include .ipynb_checkpoints notebooks. Defaults to skipping them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = extract_figures(
        root=Path(args.root).resolve(),
        out_dir=Path(args.out).resolve(),
        overwrite=args.overwrite,
        include_checkpoints=args.include_checkpoints,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a comparison table for the best single- and multi-layer models."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_PATHS = [
    ROOT / "single_layer_results/graphs/best_single_multi_model_table.svg",
    ROOT / "multi_layer_results/graphs/best_single_multi_model_table.svg",
]

BG = "#f7f4ea"
PANEL = "#fffdf7"
GRID = "#b8b1a1"
TEXT = "#1f1d1a"
SUBTLE = "#6f6659"
HEADER = "#ede6d2"


def read_summary(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle))


def avg(a: float, b: float) -> float:
    return (a + b) / 2.0


def fmt_decimal(value: float) -> str:
    return f"{value:.3f}"


def fmt_money(value: float) -> str:
    return f"${value:,.0f}"


def model_rows() -> list[list[str]]:
    single = read_summary(ROOT / "single_layer_results/overall_best_model_summary.csv")
    multi = read_summary(ROOT / "multi_layer_results/overall_best_model_summary.csv")

    single_r2 = avg(float(single["val_R2_score"]), float(single["test_R2_score"]))
    single_r = avg(float(single["val_R"]), float(single["test_R"]))
    single_rmse = avg(float(single["val_rmse"]), float(single["test_rmse"]))

    multi_r2 = avg(float(multi["val_R2_score"]), float(multi["test_R2_score"]))
    multi_r = avg(float(multi["val_R"]), float(multi["test_R"]))
    multi_rmse = avg(float(multi["val_rmse"]), float(multi["test_rmse"]))

    return [
        [
            f"Single Layer ({single['neurons']} neurons, {single['epochs']} epochs)",
            f"{fmt_decimal(single_r2)} (variance explained)",
            f"{fmt_decimal(single_r)} (correlation strength)",
            f"{fmt_money(single_rmse)} (average error in dollars)",
        ],
        [
            f"Multi Layer ({multi['layer_1_neurons']}/{multi['layer_2_neurons']} neurons, {multi['epochs']} epochs)",
            f"{fmt_decimal(multi_r2)} (variance explained)",
            f"{fmt_decimal(multi_r)} (correlation strength)",
            f"{fmt_money(multi_rmse)} (average error in dollars)",
        ],
    ]


def build_svg(rows: list[list[str]]) -> str:
    width, height = 1380, 300
    left = 28
    top = 26
    table_width = width - 56
    title_y = 40
    subtitle_y = 64
    header_y = 86
    row_h = 64
    cols = [0.30, 0.18, 0.18, 0.34]
    col_x = [left]
    for frac in cols[:-1]:
        col_x.append(col_x[-1] + table_width * frac)
    col_widths = [table_width * frac for frac in cols]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Best single and multi layer model summary table">',
        f'<rect width="{width}" height="{height}" fill="{BG}" />',
        f'<rect x="{left}" y="{top}" width="{table_width}" height="{height - 2*top}" rx="14" fill="{PANEL}" stroke="{GRID}" stroke-width="1.5" />',
        f'<text x="{width/2:.1f}" y="{title_y}" text-anchor="middle" fill="{TEXT}" font-size="28" font-weight="700" font-family="Georgia, serif">Best Single and Multi Layer Models</text>',
        f'<text x="{width/2:.1f}" y="{subtitle_y}" text-anchor="middle" fill="{SUBTLE}" font-size="14" font-family="Trebuchet MS, Arial, sans-serif">Average of validation and test metrics for each overall best model</text>',
    ]

    headers = [
        "Model",
        "R^2",
        "R",
        "RMSE",
    ]

    header_height = 42
    lines.append(
        f'<rect x="{left}" y="{header_y}" width="{table_width}" height="{header_height}" fill="{HEADER}" stroke="{GRID}" stroke-width="1.2" />'
    )

    for idx, header in enumerate(headers):
        x = col_x[idx]
        w = col_widths[idx]
        lines.append(
            f'<rect x="{x:.1f}" y="{header_y}" width="{w:.1f}" height="{header_height}" fill="none" stroke="{GRID}" stroke-width="1.2" />'
        )
        lines.append(
            f'<text x="{x + w/2:.1f}" y="{header_y + 27:.1f}" text-anchor="middle" fill="{TEXT}" font-size="18" font-weight="700" font-family="Trebuchet MS, Arial, sans-serif">{header}</text>'
        )

    start_y = header_y + header_height
    for row_idx, row in enumerate(rows):
        y = start_y + row_idx * row_h
        for col_idx, cell in enumerate(row):
            x = col_x[col_idx]
            w = col_widths[col_idx]
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{row_h:.1f}" fill="none" stroke="{GRID}" stroke-width="1.2" />'
            )
            anchor = "start" if col_idx == 0 else "middle"
            text_x = x + 16 if col_idx == 0 else x + w / 2
            lines.append(
                f'<text x="{text_x:.1f}" y="{y + 38:.1f}" text-anchor="{anchor}" fill="{TEXT}" font-size="17" font-family="Trebuchet MS, Arial, sans-serif">{cell}</text>'
            )

    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    svg = build_svg(model_rows())
    for path in OUT_PATHS:
        path.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()

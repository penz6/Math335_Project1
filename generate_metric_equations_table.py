#!/usr/bin/env python3
"""Generate an SVG table with visually formatted RMSE, R^2, and R equations."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_PATHS = [
    ROOT / "single_layer_results/graphs/metric_equations_table.svg",
    ROOT / "multi_layer_results/graphs/metric_equations_table.svg",
]

BG = "#f7f4ea"
PANEL = "#fffdf7"
GRID = "#d9d2bf"
TEXT = "#1f1d1a"
TRAIN = "#355070"
VAL = "#c1121f"
ACCENT = "#2a9d8f"


def text(x: float, y: float, content: str, *, size: int = 24, fill: str = TEXT, weight: str = "400", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" fill="{fill}" '
        f'font-size="{size}" font-weight="{weight}" '
        'font-family="Cambria Math, STIX Two Math, Times New Roman, serif">'
        f"{content}</text>"
    )


def metric_label(x: float, y: float, label: str, color: str) -> list[str]:
    return [text(x, y, label, size=28, fill=color, weight="700", anchor="middle")]


def fraction(x: float, y: float, width: float, numerator: str, denominator: str, *, size: int = 24) -> list[str]:
    mid = x + width / 2
    return [
        text(mid, y - 12, numerator, size=size, anchor="middle"),
        f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + width:.1f}" y2="{y:.1f}" stroke="{TEXT}" stroke-width="1.8" />',
        text(mid, y + 26, denominator, size=size, anchor="middle"),
    ]


def rmse_formula(x: float, y: float) -> list[str]:
    lines = [
        text(x, y, "RMSE =", size=26),
        text(x + 124, y + 6, "√", size=54),
        f'<line x1="{x + 158:.1f}" y1="{y - 30:.1f}" x2="{x + 438:.1f}" y2="{y - 30:.1f}" stroke="{TEXT}" stroke-width="1.8" />',
    ]
    lines.extend(
        fraction(
            x + 182,
            y - 4,
            232,
            "Σ (yᵢ − ŷᵢ)²",
            "n",
            size=24,
        )
    )
    return lines


def r2_formula(x: float, y: float) -> list[str]:
    lines = [text(x, y, "R² = 1 −", size=26)]
    lines.extend(
        fraction(
            x + 132,
            y - 4,
            330,
            "Σ (yᵢ − ŷᵢ)²",
            "Σ (yᵢ − ȳ)²",
            size=24,
        )
    )
    return lines


def r_formula(x: float, y: float) -> list[str]:
    lines = [text(x, y, "R =", size=26)]
    lines.extend(
        fraction(
            x + 84,
            y - 4,
            760,
            "Σ ((yᵢ − ȳ)(ŷᵢ − ŷ̄))",
            "√[Σ (yᵢ − ȳ)² Σ (ŷᵢ − ŷ̄)²]",
            size=23,
        )
    )
    return lines


def build_svg() -> str:
    width, height = 1180, 330
    left = 28
    top = 24
    row_h = 94
    metric_w = 170
    formula_w = width - 56 - metric_w

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="RMSE, R squared, and R equations">',
        f'<rect width="{width}" height="{height}" fill="{BG}" />',
        f'<rect x="{left}" y="{top}" width="{width - 56}" height="{height - 48}" rx="16" fill="{PANEL}" stroke="{GRID}" stroke-width="1.6" />',
    ]

    rows = [
        ("RMSE", TRAIN, rmse_formula),
        ("R²", ACCENT, r2_formula),
        ("R", VAL, r_formula),
    ]

    for idx, (label, color, formula_fn) in enumerate(rows):
        y = top + idx * row_h
        lines.append(f'<rect x="{left}" y="{y}" width="{metric_w}" height="{row_h}" fill="none" stroke="{GRID}" stroke-width="1.2" />')
        lines.append(f'<rect x="{left + metric_w}" y="{y}" width="{formula_w}" height="{row_h}" fill="none" stroke="{GRID}" stroke-width="1.2" />')
        lines.extend(metric_label(left + metric_w / 2, y + 56, label, color))
        lines.extend(formula_fn(left + metric_w + 22, y + 56))

    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    svg = build_svg()
    for path in OUT_PATHS:
        path.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate presentation-ready SVG graphs for multi-layer model results."""

from __future__ import annotations

import csv
import math
from collections import deque
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent

CONFIG_COLORS = {
    (10, 5): "#1b4965",
    (25, 10): "#ca6702",
    (50, 25): "#0b6e4f",
    (75, 50): "#b02e0c",
    (100, 75): "#5f0f40",
    (125, 100): "#3a5a40",
    (200, 125): "#6c757d",
    (350, 200): "#7b2cbf",
}

BG = "#f7f4ea"
PANEL = "#fffdf7"
GRID = "#d9d2bf"
AXIS = "#544b3d"
TEXT = "#1f1d1a"
SUBTLE = "#6f6659"
TRAIN = "#355070"
VAL = "#c1121f"

# read csv rows into memory
def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))

# read one numeric column
def read_single_column(path: Path) -> list[float]:
    rows = read_csv_rows(path)
    if not rows:
        return []
    column = next(iter(rows[0].keys()))
    return [float(row[column]) for row in rows]

# format whole numbers
def fmt_int(value: float) -> str:
    return f"{int(round(value)):,}"

# format short labels
def fmt_short(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M".rstrip("0").rstrip(".")
    if abs(value) >= 1000:
        return f"{value/1000:.0f}k"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")

# format metric values
def fmt_metric(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")

# format short money labels
def fmt_currency_short(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:.1f}M".rstrip("0").rstrip(".")
    if abs(value) >= 1_000:
        return f"${value/1_000:.0f}k"
    return f"${int(round(value)):,}"

# escape svg text
def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

# scale values to screen space
def scale_linear(value: float, domain: tuple[float, float], span: tuple[float, float]) -> float:
    lo, hi = domain
    start, end = span
    if hi == lo:
        return (start + end) / 2
    ratio = (value - lo) / (hi - lo)
    return start + ratio * (end - start)

# add padding to a domain
def padded_domain(values: Iterable[float], pad_fraction: float = 0.06) -> tuple[float, float]:
    values = list(values)
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        delta = abs(lo) * 0.1 or 1.0
        return lo - delta, hi + delta
    span = hi - lo
    pad = span * pad_fraction
    return lo - pad, hi + pad

# build nice axis ticks
def nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    raw_step = (hi - lo) / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    residual = raw_step / magnitude
    if residual < 1.5:
        nice_step = 1 * magnitude
    elif residual < 3:
        nice_step = 2 * magnitude
    elif residual < 7:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude
    tick_start = math.floor(lo / nice_step) * nice_step
    tick_end = math.ceil(hi / nice_step) * nice_step
    ticks = []
    value = tick_start
    while value <= tick_end + nice_step * 0.5:
        ticks.append(round(value, 10))
        value += nice_step
    return ticks

# reduce the number of ticks
def reduce_ticks(values: list[int], max_ticks: int = 8) -> list[int]:
    ordered = sorted(dict.fromkeys(values))
    if len(ordered) <= max_ticks:
        return ordered
    positions = []
    for idx in range(max_ticks):
        pos = round(idx * (len(ordered) - 1) / (max_ticks - 1))
        positions.append(ordered[pos])
    return sorted(dict.fromkeys(positions))

# build an svg path string
def build_path(points: list[tuple[float, float]]) -> str:
    return " ".join(
        [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
        + [f"L {x:.2f} {y:.2f}" for x, y in points[1:]]
    )

# compute the correlation value
def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or not xs:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)

# compute the line of best fit
def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if len(xs) != len(ys) or not xs:
        return 1.0, 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return 1.0, 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x
    return slope, intercept

# build the feature correlation data
def compute_feature_correlation_matrix(include_categorical: bool = False) -> tuple[list[str], list[list[float]]]:
    source = Path("/home/penn/Downloads/city_market_tracker.tsv000")
    sample_every = 25
    if include_categorical:
        labels = [
            "State",
            "City",
            "Type",
            "Inventory",
            "Region",
            "Homes Sold",
            "Median DOM",
            "Avg Sale/List",
            "Lag 1",
            "Lag 3",
            "Month Sin",
            "Month Cos",
            "Year",
            "Sale Price",
        ]
        categorical_cols = ["STATE", "CITY", "PROPERTY_TYPE", "REGION"]
        encoders = {col: {} for col in categorical_cols}
    else:
        labels = [
            "Inventory",
            "Homes Sold",
            "Median DOM",
            "Avg Sale/List",
            "Lag 1",
            "Lag 3",
            "Month Sin",
            "Month Cos",
            "Year",
            "Sale Price",
        ]
    lag_history: dict[str, deque[float]] = {}
    columns = [[] for _ in labels]

    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_idx, row in enumerate(reader):
            if row_idx % sample_every != 0:
                continue
            region = row.get("REGION", "")
            price_text = row.get("MEDIAN_SALE_PRICE", "")
            period_text = row.get("PERIOD_BEGIN", "")
            if not region or not price_text or not period_text:
                continue
            try:
                sale_price = float(price_text)
                inventory = float(row["INVENTORY"])
                homes_sold = float(row["HOMES_SOLD"])
                median_dom = float(row["MEDIAN_DOM"])
                avg_sale_to_list = float(row["AVG_SALE_TO_LIST"])
            except (KeyError, ValueError):
                continue

            history = lag_history.setdefault(region, deque(maxlen=3))
            if len(history) >= 3:
                year, month = int(period_text[:4]), int(period_text[5:7])
                month_sin = math.sin(2 * math.pi * month / 12)
                month_cos = math.cos(2 * math.pi * month / 12)
                if include_categorical:
                    values = [
                        float(encoders["STATE"].setdefault(row["STATE"], len(encoders["STATE"]))),
                        float(encoders["CITY"].setdefault(row["CITY"], len(encoders["CITY"]))),
                        float(encoders["PROPERTY_TYPE"].setdefault(row["PROPERTY_TYPE"], len(encoders["PROPERTY_TYPE"]))),
                        inventory,
                        float(encoders["REGION"].setdefault(region, len(encoders["REGION"]))),
                        homes_sold,
                        median_dom,
                        avg_sale_to_list,
                        history[-1],
                        history[0],
                        month_sin,
                        month_cos,
                        float(year),
                        sale_price,
                    ]
                else:
                    values = [
                        inventory,
                        homes_sold,
                        median_dom,
                        avg_sale_to_list,
                        history[-1],
                        history[0],
                        month_sin,
                        month_cos,
                        float(year),
                        sale_price,
                    ]
                for idx, value in enumerate(values):
                    columns[idx].append(value)
            history.append(sale_price)

    matrix = []
    for xs in columns:
        matrix.append([pearson_correlation(xs, ys) for ys in columns])
    return labels, matrix

# build the svg header
def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f'<rect width="{width}" height="{height}" fill="{BG}" />',
    ]

# save the svg file
def save_svg(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines + ["</svg>"]), encoding="utf-8")

# draw the chart frame
def draw_chart_frame(
    lines: list[str],
    left: float,
    top: float,
    width: float,
    height: float,
    title: str,
    subtitle: str = "",
) -> tuple[float, float, float, float]:
    lines.append(
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="18" fill="{PANEL}" stroke="#e6dec9" />'
    )
    lines.append(
        f'<text x="{left + 24:.1f}" y="{top + 34:.1f}" fill="{TEXT}" '
        'font-size="22" font-weight="700" font-family="Georgia, serif">'
        f"{escape(title)}</text>"
    )
    if subtitle:
        lines.append(
            f'<text x="{left + 24:.1f}" y="{top + 58:.1f}" fill="{SUBTLE}" '
            'font-size="13" font-family="Trebuchet MS, Arial, sans-serif">'
            f"{escape(subtitle)}</text>"
        )
    plot_left = left + 70
    plot_top = top + (82 if subtitle else 66)
    plot_width = width - 100
    plot_height = height - (130 if subtitle else 114)
    return plot_left, plot_top, plot_width, plot_height

# draw a line chart
def draw_line_chart(
    path: Path,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    series: list[dict[str, object]],
    width: int = 1180,
    height: int = 720,
    y_tick_formatter=fmt_short,
    clamp_y_min_zero: bool = False,
    footer_text: str = "",
) -> None:
    x_values = [x for item in series for x in item["x"]]  # type: ignore[index]
    y_values = [y for item in series for y in item["y"]]  # type: ignore[index]
    x_domain = padded_domain(x_values, 0.02)
    y_domain = padded_domain(y_values, 0.08)
    if clamp_y_min_zero:
        y_domain = (max(0.0, y_domain[0]), y_domain[1])
    x_ticks = reduce_ticks([int(x) for x in x_values], 8)
    y_ticks = nice_ticks(y_domain[0], y_domain[1], 6)

    lines = svg_header(width, height, title)
    plot_left, plot_top, plot_width, plot_height = draw_chart_frame(
        lines, 28, 24, width - 56, height - 48, title, subtitle
    )
    plot_right = plot_left + plot_width
    plot_bottom = plot_top + plot_height

    for tick in y_ticks:
        y = scale_linear(tick, y_domain, (plot_bottom, plot_top))
        lines.append(
            f'<line x1="{plot_left:.1f}" y1="{y:.1f}" x2="{plot_right:.1f}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="1" />'
        )
        lines.append(
            f'<text x="{plot_left - 12:.1f}" y="{y + 5:.1f}" text-anchor="end" fill="{SUBTLE}" '
            'font-size="12" font-family="Trebuchet MS, Arial, sans-serif">'
            f"{escape(y_tick_formatter(tick))}</text>"
        )

    for tick in x_ticks:
        x = scale_linear(tick, x_domain, (plot_left, plot_right))
        lines.append(
            f'<line x1="{x:.1f}" y1="{plot_top:.1f}" x2="{x:.1f}" y2="{plot_bottom:.1f}" '
            f'stroke="{GRID}" stroke-width="1" />'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{plot_bottom + 22:.1f}" text-anchor="middle" fill="{SUBTLE}" '
            'font-size="12" font-family="Trebuchet MS, Arial, sans-serif">'
            f"{tick}</text>"
        )

    lines.append(
        f'<line x1="{plot_left:.1f}" y1="{plot_bottom:.1f}" x2="{plot_right:.1f}" y2="{plot_bottom:.1f}" '
        f'stroke="{AXIS}" stroke-width="1.6" />'
    )
    lines.append(
        f'<line x1="{plot_left:.1f}" y1="{plot_top:.1f}" x2="{plot_left:.1f}" y2="{plot_bottom:.1f}" '
        f'stroke="{AXIS}" stroke-width="1.6" />'
    )

    for item in series:
        xs = item["x"]  # type: ignore[index]
        ys = item["y"]  # type: ignore[index]
        points = [
            (
                scale_linear(float(x), x_domain, (plot_left, plot_right)),
                scale_linear(float(y), y_domain, (plot_bottom, plot_top)),
            )
            for x, y in zip(xs, ys)
        ]
        color = str(item["color"])
        lines.append(
            f'<path d="{build_path(points)}" fill="none" stroke="{color}" '
            'stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" />'
        )
        for x, y in points:
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.4" fill="{color}" stroke="{PANEL}" stroke-width="1.4" />'
            )

    legend_x = plot_right - 220
    legend_y = plot_top + 8
    for idx, item in enumerate(series):
        y = legend_y + idx * 24
        color = str(item["color"])
        label = str(item["label"])
        lines.append(
            f'<line x1="{legend_x:.1f}" y1="{y:.1f}" x2="{legend_x + 24:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="4" stroke-linecap="round" />'
        )
        lines.append(
            f'<text x="{legend_x + 34:.1f}" y="{y + 4:.1f}" fill="{TEXT}" '
            'font-size="13" font-family="Trebuchet MS, Arial, sans-serif">'
            f"{escape(label)}</text>"
        )

    lines.append(
        f'<text x="{(plot_left + plot_right)/2:.1f}" y="{height - 26:.1f}" text-anchor="middle" fill="{TEXT}" '
        'font-size="14" font-family="Trebuchet MS, Arial, sans-serif">'
        f"{escape(x_label)}</text>"
    )
    if footer_text:
        lines.append(
            f'<text x="{(plot_left + plot_right)/2:.1f}" y="{height - 8:.1f}" text-anchor="middle" fill="{SUBTLE}" '
            'font-size="12" font-family="Trebuchet MS, Arial, sans-serif">'
            f"{escape(footer_text)}</text>"
        )
    lines.append(
        f'<text x="24" y="{(plot_top + plot_bottom)/2:.1f}" transform="rotate(-90 24 {(plot_top + plot_bottom)/2:.1f})" '
        f'text-anchor="middle" fill="{TEXT}" font-size="14" '
        'font-family="Trebuchet MS, Arial, sans-serif">'
        f"{escape(y_label)}</text>"
    )
    save_svg(path, lines)

# draw the combined train loss chart
def draw_combined_train_loss_chart(path: Path, histories: dict[tuple[int, int], dict[str, list[float]]]) -> None:
    series = []
    for config in sorted(histories):
        loss_values = histories[config]["loss"]
        epochs = list(range(1, len(loss_values) + 1))
        log_losses = [math.log10(max(value, 1e-12)) for value in loss_values]
        series.append(
            {
                "label": f"{config[0]}/{config[1]}",
                "x": epochs,
                "y": log_losses,
                "color": CONFIG_COLORS[config],
            }
        )
    draw_line_chart(
        path=path,
        title="Training Loss by Multi-Layer Configuration",
        subtitle="Lower is better. Log-scaled loss values to make late-epoch changes readable.",
        x_label="Epoch",
        y_label="Training Loss",
        series=series,
        y_tick_formatter=lambda tick: fmt_metric(10 ** tick, 5),
    )

# draw the simple train loss chart
def draw_easy_to_read_train_loss_chart(path: Path, histories: dict[tuple[int, int], dict[str, list[float]]]) -> None:
    width, height = 1180, 720
    title = "Easy To Read: Training Loss by Configuration"
    subtitle = "Lower is better. End labels replace the legend for poster readability."
    lines = svg_header(width, height, title)
    plot_left, plot_top, plot_width, plot_height = draw_chart_frame(
        lines, 28, 24, width - 56, height - 48, title, subtitle
    )
    plot_right = plot_left + plot_width
    plot_bottom = plot_top + plot_height

    series = []
    all_x = []
    all_y = []
    for config in sorted(histories):
        loss_values = histories[config]["loss"]
        epochs = list(range(1, len(loss_values) + 1))
        log_losses = [math.log10(max(value, 1e-12)) for value in loss_values]
        series.append((config, epochs, log_losses, CONFIG_COLORS[config]))
        all_x.extend(epochs)
        all_y.extend(log_losses)

    x_domain = padded_domain(all_x, 0.02)
    y_domain = padded_domain(all_y, 0.08)
    x_ticks = reduce_ticks([int(x) for x in all_x], 8)
    y_ticks = nice_ticks(y_domain[0], y_domain[1], 6)

    for tick in y_ticks:
        y = scale_linear(tick, y_domain, (plot_bottom, plot_top))
        lines.append(
            f'<line x1="{plot_left:.1f}" y1="{y:.1f}" x2="{plot_right:.1f}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="1" />'
        )
        lines.append(
            f'<text x="{plot_left - 12:.1f}" y="{y + 5:.1f}" text-anchor="end" fill="{SUBTLE}" '
            'font-size="15" font-family="Trebuchet MS, Arial, sans-serif">'
            f"{fmt_metric(10 ** tick, 5)}</text>"
        )

    for tick in x_ticks:
        x = scale_linear(tick, x_domain, (plot_left, plot_right))
        lines.append(
            f'<line x1="{x:.1f}" y1="{plot_top:.1f}" x2="{x:.1f}" y2="{plot_bottom:.1f}" '
            f'stroke="{GRID}" stroke-width="1" />'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{plot_bottom + 24:.1f}" text-anchor="middle" fill="{SUBTLE}" '
            'font-size="15" font-family="Trebuchet MS, Arial, sans-serif">'
            f"{tick}</text>"
        )

    lines.append(
        f'<line x1="{plot_left:.1f}" y1="{plot_bottom:.1f}" x2="{plot_right:.1f}" y2="{plot_bottom:.1f}" '
        f'stroke="{AXIS}" stroke-width="1.8" />'
    )
    lines.append(
        f'<line x1="{plot_left:.1f}" y1="{plot_top:.1f}" x2="{plot_left:.1f}" y2="{plot_bottom:.1f}" '
        f'stroke="{AXIS}" stroke-width="1.8" />'
    )

    for config, epochs, log_losses, color in series:
        points = [
            (
                scale_linear(float(x), x_domain, (plot_left, plot_right)),
                scale_linear(float(y), y_domain, (plot_bottom, plot_top)),
            )
            for x, y in zip(epochs, log_losses)
        ]
        lines.append(
            f'<path d="{build_path(points)}" fill="none" stroke="{color}" '
            'stroke-width="3.8" stroke-linecap="round" stroke-linejoin="round" />'
        )
        for x, y in points:
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.8" fill="{color}" stroke="{PANEL}" stroke-width="1.4" />'
            )

    legend_x = plot_right - 220
    legend_y = plot_top + 12
    for idx, (config, _, _, color) in enumerate(series):
        y = legend_y + idx * 24
        lines.append(
            f'<line x1="{legend_x:.1f}" y1="{y:.1f}" x2="{legend_x + 24:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="4.2" stroke-linecap="round" />'
        )
        lines.append(
            f'<circle cx="{legend_x + 12:.1f}" cy="{y:.1f}" r="4.6" fill="{color}" stroke="{PANEL}" stroke-width="1.2" />'
        )
        lines.append(
            f'<text x="{legend_x + 34:.1f}" y="{y + 5:.1f}" fill="{TEXT}" '
            'font-size="15" font-family="Trebuchet MS, Arial, sans-serif">'
            f"{config[0]}/{config[1]}</text>"
        )

    lines.append(
        f'<text x="{(plot_left + plot_right)/2:.1f}" y="{height - 26:.1f}" text-anchor="middle" fill="{TEXT}" '
        'font-size="17" font-family="Trebuchet MS, Arial, sans-serif">Epoch</text>'
    )
    lines.append(
        f'<text x="24" y="{(plot_top + plot_bottom)/2:.1f}" transform="rotate(-90 24 {(plot_top + plot_bottom)/2:.1f})" '
        f'text-anchor="middle" fill="{TEXT}" font-size="17" '
        'font-family="Trebuchet MS, Arial, sans-serif">Training Loss</text>'
    )
    save_svg(path, lines)

# draw the loss grid
def draw_loss_grid(path: Path, best_rows: list[dict[str, float]], histories: dict[tuple[int, int], dict[str, list[float]]]) -> None:
    width, height = 1400, 1250
    lines = svg_header(width, height, "Complete Multi-Layer Loss Curves")
    lines.append(
        f'<text x="42" y="46" fill="{TEXT}" font-size="28" font-weight="700" '
        'font-family="Georgia, serif">Complete Multi-Layer Loss Curves</text>'
    )

    panel_w, panel_h = 420, 255
    left_margin, top_margin = 42, 76
    x_gap, y_gap = 28, 26

    for idx, row in enumerate(best_rows):
        layer_1 = int(row["layer_1_neurons"])
        layer_2 = int(row["layer_2_neurons"])
        epochs = int(row["epochs"])
        history = histories[(layer_1, layer_2)]
        losses = history["loss"]
        val_losses = history["val_loss"]
        x_values = list(range(1, len(losses) + 1))
        log_losses = [math.log10(max(v, 1e-12)) for v in losses]
        log_val = [math.log10(max(v, 1e-12)) for v in val_losses]
        y_domain = padded_domain(log_losses + log_val, 0.08)

        col = idx % 3
        row_idx = idx // 3
        left = left_margin + col * (panel_w + x_gap)
        top = top_margin + row_idx * (panel_h + y_gap)
        plot_left, plot_top, plot_width, plot_height = draw_chart_frame(
            lines,
            left,
            top,
            panel_w,
            panel_h,
            f"{layer_1}/{layer_2} neurons",
            "",
        )
        plot_right = plot_left + plot_width
        plot_bottom = plot_top + plot_height
        x_domain = (1, len(x_values))
        selected_epoch = min(max(epochs, 1), len(x_values))
        selected_idx = selected_epoch - 1
        tick_positions = sorted(set(pos for pos in [1, len(x_values) // 2, len(x_values)] if 1 <= pos <= len(x_values)))
        y_ticks = nice_ticks(y_domain[0], y_domain[1], 5)

        for tick in y_ticks:
            y = scale_linear(tick, y_domain, (plot_bottom, plot_top))
            lines.append(
                f'<line x1="{plot_left:.1f}" y1="{y:.1f}" x2="{plot_right:.1f}" y2="{y:.1f}" '
                f'stroke="{GRID}" stroke-width="1" />'
            )
            lines.append(
                f'<text x="{plot_left - 8:.1f}" y="{y + 4:.1f}" text-anchor="end" fill="{SUBTLE}" '
                'font-size="10" font-family="Trebuchet MS, Arial, sans-serif">'
                f"{fmt_metric(10 ** tick, 5)}</text>"
            )

        for tick in tick_positions:
            x = scale_linear(tick, x_domain, (plot_left, plot_right))
            lines.append(
                f'<line x1="{x:.1f}" y1="{plot_top:.1f}" x2="{x:.1f}" y2="{plot_bottom:.1f}" '
                f'stroke="{GRID}" stroke-width="1" />'
            )
            lines.append(
                f'<text x="{x:.1f}" y="{plot_bottom + 18:.1f}" text-anchor="middle" fill="{SUBTLE}" '
                'font-size="10" font-family="Trebuchet MS, Arial, sans-serif">'
                f"{tick}</text>"
            )

        train_points = [
            (
                scale_linear(x, x_domain, (plot_left, plot_right)),
                scale_linear(y, y_domain, (plot_bottom, plot_top)),
            )
            for x, y in zip(x_values, log_losses)
        ]
        val_points = [
            (
                scale_linear(x, x_domain, (plot_left, plot_right)),
                scale_linear(y, y_domain, (plot_bottom, plot_top)),
            )
            for x, y in zip(x_values, log_val)
        ]
        lines.append(
            f'<path d="{build_path(train_points)}" fill="none" stroke="{TRAIN}" '
            'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" />'
        )
        lines.append(
            f'<path d="{build_path(val_points)}" fill="none" stroke="{VAL}" '
            'stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" />'
        )

        best_x, best_y = val_points[selected_idx]
        label_text = f"val loss {fmt_metric(val_losses[selected_idx], 5)}"
        label_x = min(best_x + 8, plot_right - 84)
        label_y = max(best_y - 10, plot_top + 12)
        lines.append(
            f'<circle cx="{best_x:.1f}" cy="{best_y:.1f}" r="5" fill="{VAL}" stroke="{PANEL}" stroke-width="1.4" />'
        )
        lines.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" fill="{VAL}" '
            'font-size="10" font-weight="700" font-family="Trebuchet MS, Arial, sans-serif">'
            f"{label_text}</text>"
        )

        lines.append(
            f'<text x="{plot_left:.1f}" y="{plot_bottom + 36:.1f}" fill="{TEXT}" '
            'font-size="10" font-family="Trebuchet MS, Arial, sans-serif">Epoch</text>'
        )
        lines.append(
            f'<text x="{plot_right - 86:.1f}" y="{plot_top + 12:.1f}" fill="{TRAIN}" '
            'font-size="10" font-family="Trebuchet MS, Arial, sans-serif">Train loss</text>'
        )
        lines.append(
            f'<text x="{plot_right - 86:.1f}" y="{plot_top + 28:.1f}" fill="{VAL}" '
            'font-size="10" font-family="Trebuchet MS, Arial, sans-serif">Validation loss</text>'
        )

    save_svg(path, lines)

# draw the best metrics panels
def draw_best_metrics_panels(path: Path, best_rows: list[dict[str, float]]) -> None:
    width, height = 1180, 720
    lines = svg_header(width, height, "Best Model Metrics by Configuration")
    lines.append(
        f'<text x="42" y="46" fill="{TEXT}" font-size="28" font-weight="700" '
        'font-family="Georgia, serif">Best Model Metrics by Configuration</text>'
    )
    lines.append(
        f'<text x="42" y="68" fill="{SUBTLE}" font-size="13" font-family="Trebuchet MS, Arial, sans-serif">'
        'RMSE and training time: lower is better. R^2: higher is better.</text>'
    )

    labels = [f"{int(item['layer_1_neurons'])}/{int(item['layer_2_neurons'])}" for item in best_rows]
    x_positions = list(range(1, len(best_rows) + 1))
    specs = [
        ("Best Test RMSE", [item["test_rmse"] for item in best_rows], "#264653"),
        ("Best Test R^2", [item["test_R2_score"] for item in best_rows], "#2a9d8f"),
        ("Best Training Time (sec)", [item["elapsed_time"] for item in best_rows], "#e76f51"),
    ]

    panel_left = 42
    panel_w = width - 84
    panel_h = 180
    top_start = 76
    gap = 18

    for idx, (title, values, color) in enumerate(specs):
        top = top_start + idx * (panel_h + gap)
        plot_left, plot_top, plot_width, plot_height = draw_chart_frame(lines, panel_left, top, panel_w, panel_h, title, "")
        plot_right = plot_left + plot_width
        plot_bottom = plot_top + plot_height
        x_domain = padded_domain(x_positions, 0.04)
        y_domain = padded_domain(values, 0.08)
        y_ticks = nice_ticks(y_domain[0], y_domain[1], 5)

        for tick in y_ticks:
            y = scale_linear(tick, y_domain, (plot_bottom, plot_top))
            lines.append(
                f'<line x1="{plot_left:.1f}" y1="{y:.1f}" x2="{plot_right:.1f}" y2="{y:.1f}" '
                f'stroke="{GRID}" stroke-width="1" />'
            )
            lines.append(
                f'<text x="{plot_left - 10:.1f}" y="{y + 4:.1f}" text-anchor="end" fill="{SUBTLE}" '
                'font-size="11" font-family="Trebuchet MS, Arial, sans-serif">'
                f"{escape(fmt_short(tick))}</text>"
            )

        for x_pos, label in zip(x_positions, labels):
            x = scale_linear(x_pos, x_domain, (plot_left, plot_right))
            lines.append(
                f'<line x1="{x:.1f}" y1="{plot_top:.1f}" x2="{x:.1f}" y2="{plot_bottom:.1f}" '
                f'stroke="{GRID}" stroke-width="1" />'
            )
            lines.append(
                f'<text x="{x:.1f}" y="{plot_bottom + 18:.1f}" text-anchor="middle" fill="{SUBTLE}" '
                'font-size="11" font-family="Trebuchet MS, Arial, sans-serif">'
                f"{escape(label)}</text>"
            )

        points = [
            (
                scale_linear(x_pos, x_domain, (plot_left, plot_right)),
                scale_linear(value, y_domain, (plot_bottom, plot_top)),
            )
            for x_pos, value in zip(x_positions, values)
        ]
        lines.append(
            f'<path d="{build_path(points)}" fill="none" stroke="{color}" '
            'stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" />'
        )
        for point_idx, (x, y) in enumerate(points):
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.8" fill="{color}" stroke="{PANEL}" stroke-width="1.3" />'
            )
            value_label = fmt_short(values[point_idx]) if "RMSE" in title or "Time" in title else fmt_metric(values[point_idx])
            lines.append(
                f'<text x="{x + 8:.1f}" y="{y - 10:.1f}" fill="{color}" '
                'font-size="10" font-family="Trebuchet MS, Arial, sans-serif">'
                f"{escape(value_label)}</text>"
            )

        lines.append(
            f'<text x="{(plot_left + plot_right)/2:.1f}" y="{plot_bottom + 38:.1f}" text-anchor="middle" fill="{TEXT}" '
            'font-size="12" font-family="Trebuchet MS, Arial, sans-serif">Layer 1 / Layer 2 Neurons</text>'
        )

    save_svg(path, lines)

# draw the validation and test panels
def draw_validation_vs_test_panels(path: Path, best_rows: list[dict[str, float]]) -> None:
    width, height = 1180, 720
    lines = svg_header(width, height, "Validation vs Test Metrics")
    lines.append(
        f'<text x="42" y="46" fill="{TEXT}" font-size="28" font-weight="700" '
        'font-family="Georgia, serif">Validation vs Test Metrics</text>'
    )
    lines.append(
        f'<text x="42" y="68" fill="{SUBTLE}" font-size="13" font-family="Trebuchet MS, Arial, sans-serif">'
        'RMSE and MAPE: lower is better. R^2: higher is better.</text>'
    )
    labels = [f"{int(item['layer_1_neurons'])}/{int(item['layer_2_neurons'])}" for item in best_rows]
    x_positions = list(range(1, len(best_rows) + 1))
    specs = [
        ("RMSE", [item["val_rmse"] for item in best_rows], [item["test_rmse"] for item in best_rows]),
        ("MAPE", [item["val_mape"] for item in best_rows], [item["test_mape"] for item in best_rows]),
        ("R^2", [item["val_R2_score"] for item in best_rows], [item["test_R2_score"] for item in best_rows]),
    ]
    panel_left = 42
    panel_w = width - 84
    panel_h = 180
    top_start = 76
    gap = 18

    for idx, (metric_name, val_values, test_values) in enumerate(specs):
        top = top_start + idx * (panel_h + gap)
        plot_left, plot_top, plot_width, plot_height = draw_chart_frame(lines, panel_left, top, panel_w, panel_h, metric_name, "")
        plot_right = plot_left + plot_width
        plot_bottom = plot_top + plot_height
        x_domain = padded_domain(x_positions, 0.04)
        y_domain = padded_domain(val_values + test_values, 0.08)
        y_ticks = nice_ticks(y_domain[0], y_domain[1], 5)

        for tick in y_ticks:
            y = scale_linear(tick, y_domain, (plot_bottom, plot_top))
            lines.append(f'<line x1="{plot_left:.1f}" y1="{y:.1f}" x2="{plot_right:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1" />')
            lines.append(f'<text x="{plot_left - 10:.1f}" y="{y + 4:.1f}" text-anchor="end" fill="{SUBTLE}" font-size="11" font-family="Trebuchet MS, Arial, sans-serif">{escape(fmt_short(tick))}</text>')

        for x_pos, label in zip(x_positions, labels):
            x = scale_linear(x_pos, x_domain, (plot_left, plot_right))
            lines.append(f'<line x1="{x:.1f}" y1="{plot_top:.1f}" x2="{x:.1f}" y2="{plot_bottom:.1f}" stroke="{GRID}" stroke-width="1" />')
            lines.append(f'<text x="{x:.1f}" y="{plot_bottom + 18:.1f}" text-anchor="middle" fill="{SUBTLE}" font-size="11" font-family="Trebuchet MS, Arial, sans-serif">{escape(label)}</text>')

        for color, values in [("#355070", val_values), ("#c1121f", test_values)]:
            points = [(scale_linear(x_pos, x_domain, (plot_left, plot_right)), scale_linear(value, y_domain, (plot_bottom, plot_top))) for x_pos, value in zip(x_positions, values)]
            lines.append(f'<path d="{build_path(points)}" fill="none" stroke="{color}" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" />')
            for x, y in points:
                lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" stroke="{PANEL}" stroke-width="1.3" />')

        lines.append(f'<line x1="{plot_right - 180:.1f}" y1="{plot_top + 12:.1f}" x2="{plot_right - 156:.1f}" y2="{plot_top + 12:.1f}" stroke="#355070" stroke-width="4" stroke-linecap="round" />')
        lines.append(f'<text x="{plot_right - 146:.1f}" y="{plot_top + 16:.1f}" fill="{TEXT}" font-size="12" font-family="Trebuchet MS, Arial, sans-serif">Validation</text>')
        lines.append(f'<line x1="{plot_right - 180:.1f}" y1="{plot_top + 34:.1f}" x2="{plot_right - 156:.1f}" y2="{plot_top + 34:.1f}" stroke="#c1121f" stroke-width="4" stroke-linecap="round" />')
        lines.append(f'<text x="{plot_right - 146:.1f}" y="{plot_top + 38:.1f}" fill="{TEXT}" font-size="12" font-family="Trebuchet MS, Arial, sans-serif">Test</text>')
        lines.append(f'<text x="{(plot_left + plot_right)/2:.1f}" y="{plot_bottom + 38:.1f}" text-anchor="middle" fill="{TEXT}" font-size="12" font-family="Trebuchet MS, Arial, sans-serif">Layer 1 / Layer 2 Neurons</text>')

    save_svg(path, lines)

# draw the feature correlation heatmap
def draw_correlation_heatmap(path: Path, include_categorical: bool = False) -> None:
    labels, corr = compute_feature_correlation_matrix(include_categorical=include_categorical)
    width, height = 980, 940
    lines = svg_header(width, height, "Correlation Heatmap")
    lines.append(
        f'<text x="42" y="46" fill="{TEXT}" font-size="28" font-weight="700" '
        'font-family="Georgia, serif">Correlation Heatmap</text>'
    )
    panel_title = "Input Feature Correlations" if include_categorical else "Continuous Feature Correlations"
    panel_left, panel_top, panel_w, panel_h = draw_chart_frame(lines, 28, 24, width - 56, height - 48, panel_title, "")
    cell_size = min((panel_w - 70) / len(labels), (panel_h - 70) / len(labels))
    grid_left = panel_left + 90
    grid_top = panel_top + 24

    for i, label in enumerate(labels):
        x = grid_left + i * cell_size + cell_size / 2
        y = grid_top + i * cell_size + cell_size / 2
        lines.append(f'<text x="{x:.1f}" y="{grid_top - 10:.1f}" text-anchor="middle" fill="{TEXT}" font-size="11" font-family="Trebuchet MS, Arial, sans-serif">{escape(label)}</text>')
        lines.append(f'<text x="{grid_left - 12:.1f}" y="{y + 4:.1f}" text-anchor="end" fill="{TEXT}" font-size="11" font-family="Trebuchet MS, Arial, sans-serif">{escape(label)}</text>')

    for row_idx, corr_row in enumerate(corr):
        for col_idx, value in enumerate(corr_row):
            x = grid_left + col_idx * cell_size
            y = grid_top + row_idx * cell_size
            intensity = int(255 - 95 * abs(value))
            fill = f"rgb({intensity},{255},{intensity})" if value >= 0 else f"rgb(255,{intensity},{intensity})"
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_size:.1f}" height="{cell_size:.1f}" fill="{fill}" stroke="{BG}" />')
            lines.append(f'<text x="{x + cell_size/2:.1f}" y="{y + cell_size/2 + 4:.1f}" text-anchor="middle" fill="{TEXT}" font-size="11" font-family="Trebuchet MS, Arial, sans-serif">{value:.2f}</text>')

    save_svg(path, lines)

# draw the sorted prediction lines
def draw_sorted_predictions(path: Path, actual: list[float], predicted: list[float], layer_1: int, layer_2: int, epochs: int) -> None:
    paired = sorted(zip(actual, predicted), key=lambda item: item[0])
    if len(paired) > 1:
        low_idx = max(0, int(len(paired) * 0.01))
        high_idx = max(low_idx + 1, int(len(paired) * 0.99))
        paired = paired[low_idx:high_idx]
    low_cutoff = paired[0][0]
    high_cutoff = paired[-1][0]
    target_points = 260
    step = max(1, len(paired) // target_points)
    sampled = paired[::step]
    if sampled[-1] != paired[-1]:
        sampled.append(paired[-1])
    indices = list(range(1, len(sampled) + 1))
    actual_sample = [round(pair[0]) for pair in sampled]
    pred_sample = [round(pair[1]) for pair in sampled]

    draw_line_chart(
        path=path,
        title="Overall Best Model: Sorted Test Predictions",
        subtitle=f"Best model: {layer_1}/{layer_2} neurons, {epochs} epochs.",
        x_label="Sorted Test Sample Index",
        y_label="Sale Price (USD)",
        series=[
            {"label": "Actual price", "x": indices, "y": actual_sample, "color": "#2a9d8f"},
            {"label": "Predicted price", "x": indices, "y": pred_sample, "color": "#d62828"},
        ],
        y_tick_formatter=fmt_currency_short,
        clamp_y_min_zero=True,
        footer_text="Test data shown: 2022-02-01 to 2026-01-01",
    )

# draw the true vs prediction scatter plot
def draw_true_vs_prediction_scatter(
    path: Path,
    actual: list[float],
    predicted: list[float],
    rmse: float,
    r2_value: float,
    layer_1: int,
    layer_2: int,
    epochs: int,
) -> None:
    pairs = list(zip(actual, predicted))
    if not pairs:
        return
    pairs.sort(key=lambda item: item[0])
    if len(pairs) > 1:
        low_idx = max(0, int(len(pairs) * 0.01))
        high_idx = max(low_idx + 1, int(len(pairs) * 0.99))
        pairs = pairs[low_idx:high_idx]
    max_points = 4000
    step = max(1, len(pairs) // max_points)
    sampled = pairs[::step]
    if sampled[-1] != pairs[-1]:
        sampled.append(pairs[-1])

    sample_actual = [pair[0] for pair in sampled]
    sample_pred = [pair[1] for pair in sampled]
    lo = min(min(sample_actual), min(sample_pred))
    hi = max(max(sample_actual), max(sample_pred))
    domain = padded_domain([lo, hi], 0.06)
    slope, intercept = linear_fit(sample_actual, sample_pred)

    width, height = 1180, 720
    title = "Overall Best Model: True vs Predicted Test Prices"
    subtitle = f"Best model: {layer_1}/{layer_2} neurons, {epochs} epochs. Lower spread around the diagonal is better."
    lines = svg_header(width, height, title)
    plot_left, plot_top, plot_width, plot_height = draw_chart_frame(
        lines, 28, 24, width - 56, height - 48, title, subtitle
    )
    plot_right = plot_left + plot_width
    plot_bottom = plot_top + plot_height
    ticks = nice_ticks(domain[0], domain[1], 6)

    for tick in ticks:
        x = scale_linear(tick, domain, (plot_left, plot_right))
        y = scale_linear(tick, domain, (plot_bottom, plot_top))
        lines.append(
            f'<line x1="{plot_left:.1f}" y1="{y:.1f}" x2="{plot_right:.1f}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="1" />'
        )
        lines.append(
            f'<line x1="{x:.1f}" y1="{plot_top:.1f}" x2="{x:.1f}" y2="{plot_bottom:.1f}" '
            f'stroke="{GRID}" stroke-width="1" />'
        )
        tick_label = fmt_currency_short(tick)
        lines.append(
            f'<text x="{plot_left - 12:.1f}" y="{y + 5:.1f}" text-anchor="end" fill="{SUBTLE}" '
            'font-size="12" font-family="Trebuchet MS, Arial, sans-serif">'
            f"{escape(tick_label)}</text>"
        )
        lines.append(
            f'<text x="{x:.1f}" y="{plot_bottom + 22:.1f}" text-anchor="middle" fill="{SUBTLE}" '
            'font-size="12" font-family="Trebuchet MS, Arial, sans-serif">'
            f"{escape(tick_label)}</text>"
        )

    lines.append(
        f'<line x1="{plot_left:.1f}" y1="{plot_bottom:.1f}" x2="{plot_right:.1f}" y2="{plot_top:.1f}" '
        'stroke="#355070" stroke-width="2.4" stroke-dasharray="8 6" opacity="0.95" />'
    )
    fit_y0 = slope * domain[0] + intercept
    fit_y1 = slope * domain[1] + intercept
    fit_points = [
        (scale_linear(domain[0], domain, (plot_left, plot_right)), scale_linear(fit_y0, domain, (plot_bottom, plot_top))),
        (scale_linear(domain[1], domain, (plot_left, plot_right)), scale_linear(fit_y1, domain, (plot_bottom, plot_top))),
    ]
    lines.append(
        f'<path d="{build_path(fit_points)}" fill="none" stroke="{VAL}" '
        'stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" />'
    )

    for x_val, y_val in sampled:
        x = scale_linear(x_val, domain, (plot_left, plot_right))
        y = scale_linear(y_val, domain, (plot_bottom, plot_top))
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="#2a9d8f" fill-opacity="0.34" stroke="none" />'
        )

    legend_x = plot_right - 245
    legend_y = plot_top + 16
    lines.append(
        f'<circle cx="{legend_x:.1f}" cy="{legend_y:.1f}" r="4.5" fill="#2a9d8f" fill-opacity="0.65" stroke="none" />'
    )
    lines.append(
        f'<text x="{legend_x + 14:.1f}" y="{legend_y + 4:.1f}" fill="{TEXT}" '
        'font-size="12" font-family="Trebuchet MS, Arial, sans-serif">Test samples</text>'
    )
    lines.append(
        f'<line x1="{legend_x - 4:.1f}" y1="{legend_y + 24:.1f}" x2="{legend_x + 20:.1f}" y2="{legend_y + 24:.1f}" '
        'stroke="#355070" stroke-width="2.4" stroke-dasharray="8 6" />'
    )
    lines.append(
        f'<text x="{legend_x + 28:.1f}" y="{legend_y + 28:.1f}" fill="{TEXT}" '
        'font-size="12" font-family="Trebuchet MS, Arial, sans-serif">Perfect prediction</text>'
    )
    lines.append(
        f'<line x1="{legend_x - 4:.1f}" y1="{legend_y + 48:.1f}" x2="{legend_x + 20:.1f}" y2="{legend_y + 48:.1f}" '
        f'stroke="{VAL}" stroke-width="2.8" />'
    )
    lines.append(
        f'<text x="{legend_x + 28:.1f}" y="{legend_y + 52:.1f}" fill="{TEXT}" '
        'font-size="12" font-family="Trebuchet MS, Arial, sans-serif">Model trend</text>'
    )

    metrics_text = f"Test RMSE: {fmt_short(rmse)}   Test R^2: {fmt_metric(r2_value, 3)}"
    lines.append(
        f'<text x="{plot_left:.1f}" y="{plot_top + 18:.1f}" fill="{SUBTLE}" '
        'font-size="12" font-family="Trebuchet MS, Arial, sans-serif">'
        f"{escape(metrics_text)}</text>"
    )
    lines.append(
        f'<text x="{(plot_left + plot_right)/2:.1f}" y="{height - 26:.1f}" text-anchor="middle" fill="{TEXT}" '
        'font-size="14" font-family="Trebuchet MS, Arial, sans-serif">True Sale Price (USD)</text>'
    )
    lines.append(
        f'<text x="24" y="{(plot_top + plot_bottom)/2:.1f}" transform="rotate(-90 24 {(plot_top + plot_bottom)/2:.1f})" '
        f'text-anchor="middle" fill="{TEXT}" font-size="14" '
        'font-family="Trebuchet MS, Arial, sans-serif">Predicted Sale Price (USD)</text>'
    )
    lines.append(
        f'<text x="{(plot_left + plot_right)/2:.1f}" y="{height - 8:.1f}" text-anchor="middle" fill="{SUBTLE}" '
        'font-size="12" font-family="Trebuchet MS, Arial, sans-serif">'
        'Test data shown: 2022-02-01 to 2026-01-01</text>'
    )
    save_svg(path, lines)

# draw the test vs validation chart
def draw_test_vs_validation_by_epoch(path: Path, rows_for_config: list[dict[str, float]], label: str) -> None:
    items = sorted(rows_for_config, key=lambda item: item["epochs"])
    epochs = [int(item["epochs"]) for item in items]
    draw_line_chart(
        path=path,
        title=f"{label} Model: Test vs Validation RMSE",
        subtitle="",
        x_label="Epoch Budget",
        y_label="RMSE",
        series=[
            {"label": "Validation RMSE", "x": epochs, "y": [item["val_rmse"] for item in items], "color": "#355070"},
            {"label": "Test RMSE", "x": epochs, "y": [item["test_rmse"] for item in items], "color": "#c1121f"},
        ],
    )

# run all the graph builders
def main() -> None:
    rows = []
    for raw in read_csv_rows(ROOT / "all_model_results.csv"):
        rows.append({key: float(value) for key, value in raw.items()})

    by_config: dict[tuple[int, int], list[dict[str, float]]] = {}
    for row in rows:
        key = (int(row["layer_1_neurons"]), int(row["layer_2_neurons"]))
        by_config.setdefault(key, []).append(row)
    best_rows = [min(items, key=lambda item: item["test_rmse"]) for _, items in sorted(by_config.items())]

    histories: dict[tuple[int, int], dict[str, list[float]]] = {}
    full_run_rows = []
    full_run_histories: dict[tuple[int, int], dict[str, list[float]]] = {}
    for row in best_rows:
        layer_1 = int(row["layer_1_neurons"])
        layer_2 = int(row["layer_2_neurons"])
        epochs = int(row["epochs"])
        history_rows = read_csv_rows(ROOT / f"layer1_{layer_1}_layer2_{layer_2}_epochs_{epochs}_history.csv")
        histories[(layer_1, layer_2)] = {
            "loss": [float(item["loss"]) for item in history_rows],
            "val_loss": [float(item["val_loss"]) for item in history_rows],
        }
    for key in sorted(by_config):
        full_run = max(by_config[key], key=lambda item: item["epochs"])
        full_epochs = int(full_run["epochs"])
        history_rows = read_csv_rows(ROOT / f"layer1_{key[0]}_layer2_{key[1]}_epochs_{full_epochs}_history.csv")
        full_run_rows.append({"layer_1_neurons": float(key[0]), "layer_2_neurons": float(key[1]), "epochs": float(full_epochs)})
        full_run_histories[key] = {
            "loss": [float(item["loss"]) for item in history_rows],
            "val_loss": [float(item["val_loss"]) for item in history_rows],
        }

    config_keys = sorted(by_config.keys())

    rmse_series = []
    r2_series = []
    val_series = []
    for key in config_keys:
        items = sorted(by_config[key], key=lambda item: item["epochs"])
        epochs = [int(item["epochs"]) for item in items]
        label = f"{key[0]}/{key[1]}"
        rmse_series.append({"label": label, "x": epochs, "y": [item["test_rmse"] for item in items], "color": CONFIG_COLORS[key]})
        r2_series.append({"label": label, "x": epochs, "y": [item["test_R2_score"] for item in items], "color": CONFIG_COLORS[key]})
        val_series.append({"label": label, "x": epochs, "y": [item["val_rmse"] for item in items], "color": CONFIG_COLORS[key]})

    draw_loss_grid(OUT_DIR / "best_loss_curves_grid.svg", full_run_rows, full_run_histories)
    draw_combined_train_loss_chart(OUT_DIR / "train_loss_by_configuration.svg", full_run_histories)
    draw_easy_to_read_train_loss_chart(OUT_DIR / "easy_to_read_train_loss_by_configuration.svg", full_run_histories)

    draw_line_chart(
        path=OUT_DIR / "test_rmse_vs_epoch_budget.svg",
        title="Test RMSE vs Epoch Budget",
        subtitle="Lower is better.",
        x_label="Epoch Budget",
        y_label="Test RMSE",
        series=rmse_series,
    )

    focused_rmse_series = []
    for key in config_keys:
        items = [item for item in sorted(by_config[key], key=lambda item: item["epochs"]) if item["epochs"] >= 50]
        focused_rmse_series.append(
            {"label": f"{key[0]}/{key[1]}", "x": [int(item["epochs"]) for item in items], "y": [item["test_rmse"] for item in items], "color": CONFIG_COLORS[key]}
        )

    draw_line_chart(
        path=OUT_DIR / "test_rmse_vs_epoch_budget_focused.svg",
        title="Test RMSE vs Epoch Budget (Focused View)",
        subtitle="Lower is better.",
        x_label="Epoch Budget",
        y_label="Test RMSE",
        series=focused_rmse_series,
    )

    zoomed_rmse_series = []
    smallest_key = min(config_keys)
    for key in config_keys:
        if key == smallest_key:
            continue
        items = [item for item in sorted(by_config[key], key=lambda item: item["epochs"]) if item["epochs"] >= 50]
        zoomed_rmse_series.append(
            {"label": f"{key[0]}/{key[1]}", "x": [int(item["epochs"]) for item in items], "y": [item["test_rmse"] for item in items], "color": CONFIG_COLORS[key]}
        )

    draw_line_chart(
        path=OUT_DIR / "test_rmse_vs_epoch_budget_zoomed.svg",
        title="Test RMSE vs Epoch Budget (Zoomed Comparison)",
        subtitle="Lower is better. Excludes the smallest model to show the competitive range more clearly.",
        x_label="Epoch Budget",
        y_label="Test RMSE",
        series=zoomed_rmse_series,
    )

    draw_line_chart(
        path=OUT_DIR / "test_r2_vs_epoch_budget.svg",
        title="Test R^2 vs Epoch Budget",
        subtitle="Higher is better.",
        x_label="Epoch Budget",
        y_label="Test R^2",
        series=r2_series,
    )

    draw_line_chart(
        path=OUT_DIR / "validation_rmse_vs_epoch_budget.svg",
        title="Validation RMSE vs Epoch Budget",
        subtitle="Lower is better.",
        x_label="Epoch Budget",
        y_label="Validation RMSE",
        series=val_series,
    )

    draw_line_chart(
        path=OUT_DIR / "testing_loss_vs_epoch_budget.svg",
        title="Testing Loss vs Epoch Budget",
        subtitle="Lower is better.",
        x_label="Epoch Budget",
        y_label="Test RMSE",
        series=rmse_series,
    )

    summary = read_csv_rows(ROOT / "overall_best_model_summary.csv")[0]
    overall_best_key = (int(summary["layer_1_neurons"]), int(summary["layer_2_neurons"]))
    draw_test_vs_validation_by_epoch(
        OUT_DIR / "testing_vs_validation_loss.svg",
        by_config[overall_best_key],
        f"{overall_best_key[0]}/{overall_best_key[1]}",
    )

    best_sorted = sorted(best_rows, key=lambda item: (item["layer_1_neurons"], item["layer_2_neurons"]))
    draw_validation_vs_test_panels(OUT_DIR / "validation_vs_test_metrics.svg", best_sorted)
    draw_best_metrics_panels(OUT_DIR / "best_model_tradeoffs_by_configuration.svg", best_sorted)
    draw_correlation_heatmap(OUT_DIR / "correlation_heatmap.svg", include_categorical=False)
    draw_correlation_heatmap(OUT_DIR / "correlation_heatmap_full_inputs.svg", include_categorical=True)

    actual = read_single_column(ROOT / "y_test.csv")
    predicted = read_single_column(ROOT / "overall_best_test_predictions.csv")
    draw_sorted_predictions(
        OUT_DIR / "overall_best_test_predictions_sorted.svg",
        actual,
        predicted,
        int(summary["layer_1_neurons"]),
        int(summary["layer_2_neurons"]),
        int(summary["epochs"]),
    )
    draw_true_vs_prediction_scatter(
        OUT_DIR / "overall_best_true_vs_prediction_test.svg",
        actual,
        predicted,
        float(summary["test_rmse"]),
        float(summary["test_R2_score"]),
        int(summary["layer_1_neurons"]),
        int(summary["layer_2_neurons"]),
        int(summary["epochs"]),
    )

    manifest_lines = [
        "Generated files:",
        "best_loss_curves_grid.svg",
        "train_loss_by_configuration.svg",
        "easy_to_read_train_loss_by_configuration.svg",
        "test_rmse_vs_epoch_budget.svg",
        "test_rmse_vs_epoch_budget_focused.svg",
        "test_rmse_vs_epoch_budget_zoomed.svg",
        "test_r2_vs_epoch_budget.svg",
        "validation_rmse_vs_epoch_budget.svg",
        "testing_loss_vs_epoch_budget.svg",
        "testing_vs_validation_loss.svg",
        "validation_vs_test_metrics.svg",
        "best_model_tradeoffs_by_configuration.svg",
        "correlation_heatmap.svg",
        "correlation_heatmap_full_inputs.svg",
        "overall_best_test_predictions_sorted.svg",
        "overall_best_true_vs_prediction_test.svg",
        "",
        "Notes:",
        "- SVG format was used for slide-friendly scaling.",
        "- Loss curves are shown on a log scale to make late-epoch behavior readable.",
        "- Testing and validation loss are shown as RMSE because per-epoch test loss was not saved during training.",
        "- The heatmap uses continuous market features only.",
        "- A second heatmap includes the full encoded model inputs.",
    ]
    (OUT_DIR / "README.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

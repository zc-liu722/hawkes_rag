from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from hawkes_rag.core import Event, HawkesParams, MultivariateHawkesProcess


def plot_alpha_heatmap(
    alpha: np.ndarray,
    *,
    labels: list[str] | None = None,
    path: str | Path | None = None,
    title: str = "Hawkes interaction matrix alpha",
) -> None:
    """Write a lightweight PNG heatmap without depending on fontconfig."""
    alpha = np.asarray(alpha, dtype=float)
    cell = 46
    margin = 90
    width = margin + cell * alpha.shape[1] + 28
    height = margin + cell * alpha.shape[0] + 28
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((16, 14), title, fill=(20, 20, 20))

    max_value = float(np.max(alpha)) if alpha.size else 1.0
    max_value = max(max_value, 1e-12)
    for i in range(alpha.shape[0]):
        for j in range(alpha.shape[1]):
            color = _magma_like(float(alpha[i, j]) / max_value)
            x0 = margin + j * cell
            y0 = margin + i * cell
            draw.rectangle([x0, y0, x0 + cell - 2, y0 + cell - 2], fill=color)
    if labels is not None and len(labels) == alpha.shape[0] and len(labels) <= 16:
        for idx, label in enumerate(labels):
            short = label[:10]
            draw.text((margin + idx * cell + 2, margin - 22), short, fill=(30, 30, 30))
            draw.text((10, margin + idx * cell + 12), short, fill=(30, 30, 30))
    _save_or_show(image, path)


def plot_lambda_curve(
    params: HawkesParams,
    events: list[Event],
    memory_id: int,
    *,
    horizon: float,
    path: str | Path | None = None,
    title: str | None = None,
    n_points: int = 300,
) -> None:
    process = MultivariateHawkesProcess(params)
    times = np.linspace(0.0, horizon, n_points)
    lambdas = np.array([process.intensity(memory_id, t, events) for t in times])
    width, height = 900, 420
    left, right, top, bottom = 70, 24, 50, 54
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((16, 14), title or f"Intensity curve for memory {memory_id}", fill=(20, 20, 20))

    plot_w = width - left - right
    plot_h = height - top - bottom
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline=(210, 210, 210))

    y_min = float(np.min(lambdas))
    y_max = float(np.max(lambdas))
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0
    event_times = [event.time for event in events if event.memory_id == memory_id]
    for event_time in event_times:
        if 0 <= event_time <= horizon:
            x = left + int(plot_w * (event_time / horizon))
            draw.line([x, top, x, top + plot_h], fill=(252, 165, 165), width=1)

    points = []
    for t, lam in zip(times, lambdas):
        x = left + int(plot_w * (t / horizon))
        y = top + plot_h - int(plot_h * ((float(lam) - y_min) / (y_max - y_min)))
        points.append((x, y))
    if len(points) >= 2:
        draw.line(points, fill=(37, 99, 235), width=3)
    draw.text((left, height - 34), "time", fill=(60, 60, 60))
    draw.text((12, top + 4), "lambda(t)", fill=(60, 60, 60))
    draw.text((left, top + plot_h + 8), f"0", fill=(80, 80, 80))
    draw.text((left + plot_w - 44, top + plot_h + 8), f"{horizon:g}", fill=(80, 80, 80))
    draw.text((left + plot_w + 4, top), f"{y_max:.2f}", fill=(80, 80, 80))
    draw.text((left + plot_w + 4, top + plot_h - 12), f"{y_min:.2f}", fill=(80, 80, 80))
    _save_or_show(image, path)


def _magma_like(value: float) -> tuple[int, int, int]:
    value = min(1.0, max(0.0, value))
    stops = [
        (0.0, (13, 8, 36)),
        (0.35, (115, 31, 129)),
        (0.65, (219, 73, 89)),
        (1.0, (252, 253, 191)),
    ]
    for (x0, c0), (x1, c1) in zip(stops, stops[1:]):
        if x0 <= value <= x1:
            ratio = (value - x0) / (x1 - x0)
            return tuple(int(a + ratio * (b - a)) for a, b in zip(c0, c1))
    return stops[-1][1]


def _save_or_show(image: Image.Image, path: str | Path | None) -> None:
    if path is None:
        image.show()
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)

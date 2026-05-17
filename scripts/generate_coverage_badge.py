#!/usr/bin/env python3
"""Generate a small local SVG coverage badge from coverage.py XML output."""

from __future__ import annotations

import argparse
import html
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage_xml", type=Path, nargs="?", default=Path("coverage.xml"))
    parser.add_argument("output_svg", type=Path, nargs="?", default=Path("docs/assets/coverage.svg"))
    args = parser.parse_args()

    percent = coverage_percent(args.coverage_xml)
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text(
        render_badge("coverage", f"{percent:.0f}%", color_for_percent(percent)), encoding="utf-8"
    )


def coverage_percent(path: Path) -> float:
    root = ET.parse(path).getroot()
    line_rate = root.attrib.get("line-rate")
    if line_rate is not None:
        return float(line_rate) * 100.0

    lines_valid = int(root.attrib["lines-valid"])
    lines_covered = int(root.attrib["lines-covered"])
    if lines_valid == 0:
        return 100.0
    return lines_covered / lines_valid * 100.0


def color_for_percent(percent: float) -> str:
    if percent >= 90.0:
        return "#4c1"
    if percent >= 80.0:
        return "#97ca00"
    if percent >= 70.0:
        return "#a4a61d"
    if percent >= 60.0:
        return "#dfb317"
    return "#e05d44"


def render_badge(label: str, message: str, color: str) -> str:
    label = html.escape(label)
    message = html.escape(message)
    label_width = max(50, len(label) * 7 + 10)
    message_width = max(36, len(message) * 7 + 10)
    width = label_width + message_width
    label_center = label_width / 2
    message_center = label_width + message_width / 2
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20" role="img"
  aria-label="{label}: {message}">
  <title>{label}: {message}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r">
    <rect width="{width}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_width}" height="20" fill="#555"/>
    <rect x="{label_width}" width="{message_width}" height="20" fill="{color}"/>
    <rect width="{width}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">
    <text x="{label_center:.1f}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{label_center:.1f}" y="14">{label}</text>
    <text x="{message_center:.1f}" y="15" fill="#010101" fill-opacity=".3">{message}</text>
    <text x="{message_center:.1f}" y="14">{message}</text>
  </g>
</svg>
"""


if __name__ == "__main__":
    main()

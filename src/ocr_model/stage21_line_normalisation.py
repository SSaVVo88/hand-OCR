"""
Stage 2.1.4
Line Normalisation
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import json
import numpy as np

@dataclass(slots=True)
class LineNormalizationConfig:
	target_height: int = 128
	target_baseline_ratio: float = 0.78  # fraction of height (0-1)
	max_width: Optional[int] = 2048
	horizontal_padding: int = 24
	binarize_threshold: int = 180
	despeckle_kernel: int = 3
	min_ink_ratio: float = 0.002
	baseline_blur: int = 7
	background_value: int = 255

def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
	if image.ndim == 2:
		return image
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def _load_line(path: Path) -> np.ndarray:
	image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
	if image is None:
		raise FileNotFoundError(f"Unable to read line image: {path}")
	return image

def _trim_whitespace(gray: np.ndarray, threshold: int) -> np.ndarray:
	_, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
	rows = binary.sum(axis=1)
	cols = binary.sum(axis=0)
	if rows.max() == 0 or cols.max() == 0:
		return gray
	top = np.argmax(rows > 0)
	bottom = len(rows) - np.argmax(rows[::-1] > 0)
	left = np.argmax(cols > 0)
	right = len(cols) - np.argmax(cols[::-1] > 0)
	return gray[top:bottom, left:right]

def _resize_to_height(gray: np.ndarray, config: LineNormalizationConfig) -> np.ndarray:
	h, w = gray.shape[:2]
	if h == 0 or w == 0:
		return np.full((config.target_height, config.target_height), config.background_value, dtype=np.uint8)
	scale = config.target_height / h
	new_w = max(1, int(w * scale))
	resized = cv2.resize(gray, (new_w, config.target_height), interpolation=cv2.INTER_CUBIC)
	if config.max_width and new_w > config.max_width:
		resized = cv2.resize(resized, (config.max_width, config.target_height), interpolation=cv2.INTER_AREA)
	return resized

def _estimate_baseline_ratio(gray: np.ndarray, config: LineNormalizationConfig) -> float:
	blurred = cv2.GaussianBlur(gray, (config.despeckle_kernel | 1, config.despeckle_kernel | 1), 0)
	_, binary = cv2.threshold(blurred, config.binarize_threshold, 255, cv2.THRESH_BINARY_INV)
	ink = binary.sum()
	if ink == 0 or ink < config.min_ink_ratio * binary.size * 255:
		return config.target_baseline_ratio
	row_sums = binary.sum(axis=1).astype(np.float64)
	if row_sums.sum() == 0:
		return config.target_baseline_ratio
	kernel = config.baseline_blur if config.baseline_blur % 2 == 1 else config.baseline_blur + 1
	row_sums = cv2.GaussianBlur(row_sums.reshape(-1, 1), (kernel, 1), 0).ravel()
	indices = np.arange(len(row_sums), dtype=np.float64)
	baseline_row = float(np.dot(indices, row_sums) / (row_sums.sum() + 1e-6))
	return max(0.0, min(1.0, baseline_row / (len(row_sums) - 1 + 1e-6)))

def _shift_baseline(gray: np.ndarray, baseline_ratio: float, config: LineNormalizationConfig) -> tuple[np.ndarray, int]:
	h, w = gray.shape[:2]
	target_idx = int(config.target_baseline_ratio * (h - 1))
	current_idx = int(baseline_ratio * (h - 1))
	shift = target_idx - current_idx
	canvas = np.full_like(gray, config.background_value)
	if shift >= 0:
		canvas[shift:, :] = gray[: h - shift, :]
	else:
		canvas[: h + shift, :] = gray[-shift:, :]
	return canvas, shift

def _pad_horizontal(gray: np.ndarray, config: LineNormalizationConfig) -> np.ndarray:
	return cv2.copyMakeBorder(
		gray,
		config.horizontal_padding,
		config.horizontal_padding,
		config.horizontal_padding,
		config.horizontal_padding,
		cv2.BORDER_CONSTANT,
		value=config.background_value,
	)

def normalize_line(
	line_path: Path | str,
	source_root: Path,
	output_root: Path,
	config: LineNormalizationConfig,
) -> dict:
	line_path = Path(line_path)
	relative_parent = line_path.relative_to(source_root).parent
	out_dir = output_root / relative_parent
	out_dir.mkdir(parents=True, exist_ok=True)
	gray = _ensure_grayscale(_load_line(line_path))
	clipped = _trim_whitespace(gray, config.binarize_threshold)
	resized = _resize_to_height(clipped, config)
	baseline_ratio = _estimate_baseline_ratio(resized, config)
	baseline_aligned, shift_px = _shift_baseline(resized, baseline_ratio, config)
	padded = _pad_horizontal(baseline_aligned, config)
	output_path = out_dir / f"{line_path.stem}_norm.png"
	cv2.imwrite(str(output_path), padded)
	metadata = {
		"source": str(line_path.resolve()),
		"output": str(output_path.resolve()),
		"baseline_ratio_before": baseline_ratio,
		"shift_px": shift_px,
		"config": asdict(config),
	}
	return metadata

def batch_normalize_lines(
	input_root: Path | str,
	output_root: Path | str,
	pattern: str = "**/*.png",
	config: LineNormalizationConfig | None = None,
	manifests_dir: Path | None = None,
) -> Iterator[dict]:
	input_root = Path(input_root)
	if not input_root.exists():
		raise FileNotFoundError(f"Input lines directory missing: {input_root}")
	output_root = Path(output_root)
	config = config or LineNormalizationConfig()
	manifests_dir = Path(manifests_dir) if manifests_dir else None
	if manifests_dir:
		manifests_dir.mkdir(parents=True, exist_ok=True)
	for line_path in sorted(input_root.glob(pattern)):
		if line_path.is_dir():
			continue
		meta = normalize_line(line_path, input_root, output_root, config)
		if manifests_dir:
			relative_parent = line_path.relative_to(input_root).parent
			manifest_dir = manifests_dir / relative_parent
			manifest_dir.mkdir(parents=True, exist_ok=True)
			manifest_path = manifest_dir / f"{line_path.stem}_norm.json"
			with open(manifest_path, "w", encoding="utf-8") as fh:
				json.dump(meta, fh, indent=2)
		yield meta

def _parse_args():
	import argparse

	parser = argparse.ArgumentParser(description="Stage 2.1.4 line normalisation")
	parser.add_argument("input_root", type=Path, help="Directory with raw line crops (e.g., data/lines)")
	parser.add_argument("output_root", type=Path, help="Destination for normalised lines (e.g., data/lines_normalized)")
	parser.add_argument("--pattern", default="**/*.png", help="Glob to select subset of line files")
	parser.add_argument("--manifests-dir", type=Path, help="Optional directory to store per-line metadata JSON")
	parser.add_argument("--target-height", type=int, help="Height in pixels for normalized lines")
	parser.add_argument("--target-baseline-ratio", type=float, help="Baseline location expressed as fraction of height")
	parser.add_argument("--max-width", type=int, help="Clamp width after scaling")
	parser.add_argument("--horizontal-padding", type=int, help="Padding (px) to add on all sides")
	parser.add_argument("--binarize-threshold", type=int, help="Threshold used for whitespace trimming and baseline detection")
	parser.add_argument("--despeckle-kernel", type=int, help="Kernel size for Gaussian blur before baseline estimation")
	parser.add_argument("--baseline-blur", type=int, help="Kernel width for smoothing projection profile")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = _parse_args()
	config_kwargs = {
		"target_height": args.target_height,
		"target_baseline_ratio": args.target_baseline_ratio,
		"max_width": args.max_width,
		"horizontal_padding": args.horizontal_padding,
		"binarize_threshold": args.binarize_threshold,
		"despeckle_kernel": args.despeckle_kernel,
		"baseline_blur": args.baseline_blur,
	}
	config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}
	config = LineNormalizationConfig(**config_kwargs)
	for meta in batch_normalize_lines(
		args.input_root,
		args.output_root,
		pattern=args.pattern,
		config=config,
		manifests_dir=args.manifests_dir,
	):
		print(f"[STAGE 2.1.4] Normalized {meta['source']} -> {meta['output']}")

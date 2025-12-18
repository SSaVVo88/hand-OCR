"""
Stage 2.1.2
Page-Level Preprocessing

This module consumes the raw PNG pages produced by
src/ocr_model/stage21_pdf_to_pages.py and emits contrast-enhanced,
deskewed, and tightly cropped versions that downstream stages (line
segmentation, normalization, augmentation) can rely on.

The pipeline performs the following high-level steps:
1. Load a page image and convert it to grayscale.
2. Apply gentle denoising plus contrast-limited adaptive histogram
   equalisation (CLAHE) to stabilise stroke visibility.
3. Produce a high-quality binary mask using adaptive thresholding.
4. Estimate global skew and rotate the page so text baselines are
   horizontal.
5. Detect the text bounding box and crop extra scanner margins while
   keeping a configurable safety margin.
6. Normalize size to a fixed height (matching typical A4 @ 300 DPI) and
   pad to avoid clipping future augmentations.

Each processed page is saved next to the raw data using the suffix
"_preproc.png" and a JSON sidecar summarising the preprocessing
metadata (rotation angle, crop box, etc.).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import json, pathlib


@dataclass(slots=True)
class PreprocessConfig:
	"""Tunable knobs for page preprocessing."""

	target_height: int = 3300
	max_width: Optional[int] = None
	padding: int = 32
	margin: int = 18
	bilateral_d: int = 9
	bilateral_sigma_color: int = 75
	bilateral_sigma_space: int = 75
	clahe_clip: float = 2.0
	clahe_grid: int = 8
	adaptive_block_size: int = 45
	adaptive_c: int = 15
	min_area_ratio: float = 0.005
	min_contour_points: int = 128
	row_density_threshold: float = 0.05
	col_density_threshold: float = 0.02
	min_projection_height: int = 80
	min_projection_width: int = 80
	ignore_top_ratio: float = 0.02
	ignore_bottom_ratio: float = 0.02
	ignore_left_ratio: float = 0.01
	ignore_right_ratio: float = 0.01
	crop_mode: str = "static"  # "static" or "adaptive"
	max_auto_trim_ratio: float = 0.2
	auto_row_threshold: float = 0.004
	auto_col_threshold: float = 0.003
	output_grayscale: bool = True


def _ensure_image(image_path: Path) -> np.ndarray:
	image = cv2.imread(str(image_path))
	if image is None:
		raise FileNotFoundError(f"Unable to read image: {image_path}")
	return image


def _enhance_contrast(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	filtered = cv2.bilateralFilter(
		gray,
		config.bilateral_d,
		config.bilateral_sigma_color,
		config.bilateral_sigma_space,
	)
	clahe = cv2.createCLAHE(clipLimit=config.clahe_clip, tileGridSize=(config.clahe_grid, config.clahe_grid))
	enhanced = clahe.apply(filtered)
	return enhanced


def _threshold(enhanced_gray: np.ndarray, config: PreprocessConfig) -> np.ndarray:
	block_size = config.adaptive_block_size
	if block_size % 2 == 0:
		block_size += 1
	thresh = cv2.adaptiveThreshold(
		enhanced_gray,
		255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY_INV,
		block_size,
		config.adaptive_c,
	)
	return thresh


def _estimate_skew(binary_mask: np.ndarray, config: PreprocessConfig) -> float:
	coords = cv2.findNonZero(binary_mask)
	if coords is None or len(coords) < config.min_contour_points:
		return 0.0
	rect = cv2.minAreaRect(coords)
	angle = rect[-1]
	if angle < -45:
		angle = angle + 90
	return -angle


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
	if abs(angle) < 0.1:
		return image
	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
	cos = abs(matrix[0, 0])
	sin = abs(matrix[0, 1])
	new_w = int((h * sin) + (w * cos))
	new_h = int((h * cos) + (w * sin))
	matrix[0, 2] += (new_w / 2) - center[0]
	matrix[1, 2] += (new_h / 2) - center[1]
	return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _ensure_portrait(image: np.ndarray) -> tuple[np.ndarray, float]:
	"""Rotate by 90° counter-clockwise if the page is wider than tall."""
	h, w = image.shape[:2]
	if h >= w:
		return image, 0.0
	return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), 90.0


def _count_blank_run(values: np.ndarray, threshold: float, limit: int) -> int:
	count = 0
	for val in values:
		if val <= threshold and count < limit:
			count += 1
		else:
			break
	return count


def _compute_trim_pixels(mask: np.ndarray, config: PreprocessConfig) -> tuple[int, int, int, int]:
	h, w = mask.shape
	static_top = int(h * config.ignore_top_ratio)
	static_bottom = int(h * config.ignore_bottom_ratio)
	static_left = int(w * config.ignore_left_ratio)
	static_right = int(w * config.ignore_right_ratio)

	if config.crop_mode != "adaptive":
		return static_top, static_bottom, static_left, static_right

	row_counts = mask.sum(axis=1) / 255.0
	col_counts = mask.sum(axis=0) / 255.0
	row_threshold = config.auto_row_threshold * w
	col_threshold = config.auto_col_threshold * h
	row_limit = max(1, int(h * config.max_auto_trim_ratio))
	col_limit = max(1, int(w * config.max_auto_trim_ratio))

	auto_top = _count_blank_run(row_counts, row_threshold, row_limit)
	auto_bottom = _count_blank_run(row_counts[::-1], row_threshold, row_limit)
	auto_left = _count_blank_run(col_counts, col_threshold, col_limit)
	auto_right = _count_blank_run(col_counts[::-1], col_threshold, col_limit)

	return (
		min(h, max(static_top, auto_top)),
		min(h, max(static_bottom, auto_bottom)),
		min(w, max(static_left, auto_left)),
		min(w, max(static_right, auto_right)),
	)


def _apply_edge_trims(mask: np.ndarray, config: PreprocessConfig) -> np.ndarray:
	trimmed = mask.copy()
	h, w = trimmed.shape
	top, bottom, left, right = _compute_trim_pixels(trimmed, config)
	if top > 0:
		trimmed[:top, :] = 0
	if bottom > 0:
		trimmed[h - bottom :, :] = 0
	if left > 0:
		trimmed[:, :left] = 0
	if right > 0:
		trimmed[:, w - right :] = 0
	return trimmed


def _bounding_box(binary_mask: np.ndarray, config: PreprocessConfig) -> tuple[int, int, int, int]:
	mask = _apply_edge_trims(binary_mask, config)
	h, w = mask.shape
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return (0, 0, w, h)
	filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > config.min_area_ratio * mask.size]
	target = filtered if filtered else contours
	x, y, bw, bh = cv2.boundingRect(np.vstack(target))
	return (x, y, bw, bh)


def _refine_bbox_with_projection(
	binary_mask: np.ndarray,
	bbox: tuple[int, int, int, int],
	config: PreprocessConfig,
) -> tuple[int, int, int, int]:
	x, y, w, h = bbox
	if w < config.min_projection_width or h < config.min_projection_height:
		return bbox
	roi = binary_mask[y : y + h, x : x + w]
	if roi.size == 0:
		return bbox
	row_counts = roi.sum(axis=1) / 255.0
	col_counts = roi.sum(axis=0) / 255.0
	row_threshold = config.row_density_threshold * w
	col_threshold = config.col_density_threshold * h

	top = 0
	for idx, val in enumerate(row_counts):
		if val > row_threshold:
			top = idx
			break

	bottom = 0
	for idx, val in enumerate(row_counts[::-1]):
		if val > row_threshold:
			bottom = idx
			break

	left = 0
	for idx, val in enumerate(col_counts):
		if val > col_threshold:
			left = idx
			break

	right = 0
	for idx, val in enumerate(col_counts[::-1]):
		if val > col_threshold:
			right = idx
			break

	new_x = x + left
	new_y = y + top
	new_w = w - left - right
	new_h = h - top - bottom
	if new_w <= 0 or new_h <= 0:
		return bbox
	return (new_x, new_y, new_w, new_h)


def _crop_with_margin(image: np.ndarray, bbox: tuple[int, int, int, int], config: PreprocessConfig) -> np.ndarray:
	x, y, w, h = bbox
	h_img, w_img = image.shape[:2]
	x0 = max(x - config.margin, 0)
	y0 = max(y - config.margin, 0)
	x1 = min(x + w + config.margin, w_img)
	y1 = min(y + h + config.margin, h_img)
	return image[y0:y1, x0:x1]


def _resize_and_pad(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
	h, w = image.shape[:2]
	scale = config.target_height / h
	new_w = int(w * scale)
	resized = cv2.resize(image, (new_w, config.target_height), interpolation=cv2.INTER_CUBIC)
	if config.max_width and new_w > config.max_width:
		resized = cv2.resize(resized, (config.max_width, config.target_height), interpolation=cv2.INTER_AREA)
	padded = cv2.copyMakeBorder(
		resized,
		config.padding,
		config.padding,
		config.padding,
		config.padding,
		cv2.BORDER_CONSTANT,
		value=[255, 255, 255],
	)
	return padded


def preprocess_page_image(
	image_path: Path | str,
	output_dir: Path | str,
	config: PreprocessConfig | None = None,
) -> dict:
	"""Preprocess a single page image and return metadata."""

	config = config or PreprocessConfig()
	image_path = Path(image_path)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	image = _ensure_image(image_path)
	enhanced = _enhance_contrast(image, config)
	binary = _threshold(enhanced, config)
	angle = _estimate_skew(binary, config)
	rotated = _rotate_image(image, angle)
	rotated, orientation_adjust = _ensure_portrait(rotated)
	total_rotation = angle + orientation_adjust
	enhanced_rot = _enhance_contrast(rotated, config)
	binary_rot = _threshold(enhanced_rot, config)
	bbox = _bounding_box(binary_rot, config)
	bbox = _refine_bbox_with_projection(binary_rot, bbox, config)
	cropped = _crop_with_margin(rotated, bbox, config)
	output_image = _resize_and_pad(cropped, config)
	if config.output_grayscale and output_image.ndim == 3:
		output_to_save = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
	else:
		output_to_save = output_image

	stem = image_path.stem + "_preproc"
	output_path = output_dir / f"{stem}.png"
	cv2.imwrite(str(output_path), output_to_save)

	metadata = {
		"source": str(image_path.resolve()),
		"output": str(output_path.resolve()),
		"rotation_deg": total_rotation,
		"deskew_deg": angle,
		"orientation_adjust_deg": orientation_adjust,
		"crop_mode": config.crop_mode,
		"bbox": {
			"x": int(bbox[0]),
			"y": int(bbox[1]),
			"w": int(bbox[2]),
			"h": int(bbox[3]),
		},
		"config": asdict(config),
	}

	with open(output_dir / f"{stem}.json", "w", encoding="utf-8") as fh:
		json.dump(metadata, fh, indent=2)

	return metadata


def batch_preprocess(
	pages_dir: Path | str,
	output_dir: Path | str,
	pattern: str = "*.png",
	config: PreprocessConfig | None = None,
) -> Iterator[dict]:
	pages_dir = Path(pages_dir)
	if not pages_dir.exists():
		raise FileNotFoundError(f"Pages directory missing: {pages_dir}")
	config = config or PreprocessConfig()

	for image_path in sorted(pages_dir.glob(pattern)):
		yield preprocess_page_image(image_path, output_dir, config=config)


def _parse_args() -> tuple[Path, Path, str]:
	import argparse

	parser = argparse.ArgumentParser(description="Stage 2.1.2 page preprocessing")
	parser.add_argument("pages_dir", type=Path, help="Directory with raw page PNGs (data/pages)")
	parser.add_argument("output_dir", type=Path, help="Destination for preprocessed pages")
	parser.add_argument(
		"--crop-mode",
		choices=["static", "adaptive"],
		default="static",
		help="Edge trimming strategy before contour detection",
	)
	parser.add_argument("--ignore-top-ratio", type=float, help="Fraction of height to always trim from top")
	parser.add_argument("--ignore-bottom-ratio", type=float, help="Fraction of height to always trim from bottom")
	parser.add_argument("--ignore-left-ratio", type=float, help="Fraction of width to always trim from left")
	parser.add_argument("--ignore-right-ratio", type=float, help="Fraction of width to always trim from right")
	parser.add_argument("--margin", type=int, help="Safety padding (pixels) added back after cropping")
	parser.add_argument("--row-density-threshold", type=float, help="Row density threshold used for projection trimming")
	parser.add_argument("--col-density-threshold", type=float, help="Column density threshold used for projection trimming")
	parser.add_argument("--auto-row-threshold", type=float, help="Row density threshold for adaptive trimming mode")
	parser.add_argument("--auto-col-threshold", type=float, help="Column density threshold for adaptive trimming mode")
	parser.add_argument("--max-auto-trim-ratio", type=float, help="Max fraction of dimension auto trim can remove")
	parser.add_argument(
		"--pattern",
		default="*.png",
		help="Glob for selecting subset of images (default: *.png)",
	)
	args = parser.parse_args()
	return args




if __name__ == "__main__":
	args = _parse_args()
	config_kwargs = {"crop_mode": args.crop_mode}
	optional_fields = {
		"ignore_top_ratio": args.ignore_top_ratio,
		"ignore_bottom_ratio": args.ignore_bottom_ratio,
		"ignore_left_ratio": args.ignore_left_ratio,
		"ignore_right_ratio": args.ignore_right_ratio,
		"margin": args.margin,
		"row_density_threshold": args.row_density_threshold,
		"col_density_threshold": args.col_density_threshold,
		"auto_row_threshold": args.auto_row_threshold,
		"auto_col_threshold": args.auto_col_threshold,
		"max_auto_trim_ratio": args.max_auto_trim_ratio,
	}
	for key, value in optional_fields.items():
		if value is not None:
			config_kwargs[key] = value
	config = PreprocessConfig(**config_kwargs)
	for meta in batch_preprocess(args.pages_dir, args.output_dir, args.pattern, config=config):
		print(
			f"[STAGE 2.1.2] Processed {meta['source']} -> {meta['output']} "
			f"(rotation={meta['rotation_deg']:.2f}°, deskew={meta['deskew_deg']:.2f}°)"
		)
    # Use this to test loading metadata
	#meta = json.loads(pathlib.Path("data/pages_preproc/Adobe Scan 12 gru 2025_page_008_preproc.json").read_text())
	#print(meta["config"]["ignore_top_ratio"], meta["crop_mode"])

"""
Stage 2.1.3
Line Segmentation

Consumes preprocessed page images from Stage 2.1.2 and extracts ordered
line crops that downstream normalization/labeling stages can consume.

Pipeline synopsis:
1. Load grayscale page from data/pages_preproc/ and build a clean binary mask.
2. Compute a smoothed horizontal projection profile to locate text bands.
3. Convert the projection into contiguous line regions, merging tiny gaps and
   discarding noise.
4. Crop each region (with configurable padding), normalize height, and save
   to data/lines/<page_stem>/line_###.png.
5. Persist a JSON manifest capturing line bounding boxes, order, and config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import json


@dataclass(slots=True)
class LineSegmentationConfig:
	'''Parameters will be saved in JSON'''
	blur_kernel: int = 5
	adaptive_block_size: int = 35
	adaptive_c: int = 10
	projection_threshold: float = 0.12
	min_line_height: int = 32
	merge_gap: int = 18
	pad_top: int = 6
	pad_bottom: int = 8
	target_height: int = 256
	horizontal_padding: int = 24
	background_value: int = 255
	max_lines: Optional[int] = None
	debug: bool = False
	# New parameters for improved segmentation
	max_line_height: int = 150  # Auto-split regions taller than this
	min_ink_ratio: float = 0.015  # Filter lines with less ink (noise/empty)
	split_threshold_factor: float = 0.7  # Relative threshold for splitting tall regions
	enable_auto_split: bool = True  # Enable automatic splitting of oversized lines

#Ensure greyscale for each image (skipable if we used my previous script)
def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
	if image.ndim == 2:
		return image
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Loading image 
def _load_image(image_path: Path) -> np.ndarray:
	image = cv2.imread(str(image_path))
	if image is None:
		raise FileNotFoundError(f"Unable to read image: {image_path}")
	return image

#Recompute or reuse page binary mask 
def _binarize(gray: np.ndarray, config: LineSegmentationConfig) -> np.ndarray:
	block = config.adaptive_block_size if config.adaptive_block_size % 2 == 1 else config.adaptive_block_size + 1
	return cv2.adaptiveThreshold(
		gray,
		255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY_INV,
		block,
		config.adaptive_c,
	)

#Convert the mask into a normalized, smoothed horizontal projection profile.
def _projection_profile(binary_mask: np.ndarray, config: LineSegmentationConfig) -> np.ndarray:
	proj = binary_mask.sum(axis=1).astype(np.float32)
	proj = proj / (proj.max() + 1e-6)
	kernel = config.blur_kernel if config.blur_kernel % 2 == 1 else config.blur_kernel + 1
	proj = cv2.GaussianBlur(proj, (kernel, 1), 0)
	return proj

#Merge adjacent line regions separated by gaps smaller than the threshold.
def _region_gaps(regions: List[Tuple[int, int]], gap_threshold: int) -> List[Tuple[int, int]]:
	if not regions:
		return []
	merged: List[Tuple[int, int]] = [regions[0]]
	for start, end in regions[1:]:
		prev_start, prev_end = merged[-1]
		if start - prev_end <= gap_threshold:
			merged[-1] = (prev_start, max(prev_end, end))
		else:
			merged.append((start, end))
	return merged

#Turn the projection profile into candidate line spans using thresholding and merging.
def _detect_line_regions(projection: np.ndarray, config: LineSegmentationConfig) -> List[Tuple[int, int]]:
	regions: List[Tuple[int, int]] = []
	in_line = False
	start = 0
	for idx, value in enumerate(projection):
		if not in_line and value >= config.projection_threshold:
			in_line = True
			start = idx
		elif in_line and value < config.projection_threshold:
			end = idx
			if end - start >= config.min_line_height:
				regions.append((start, end))
			in_line = False
	if in_line:
		regions.append((start, len(projection) - 1))
	regions = _region_gaps(regions, config.merge_gap)
	return regions


def _split_tall_region(
	projection: np.ndarray,
	region: Tuple[int, int],
	config: LineSegmentationConfig,
) -> List[Tuple[int, int]]:
	"""Recursively split a region that is taller than max_line_height.
	
	Uses a lower threshold to find internal gaps within the oversized region.
	"""
	start, end = region
	height = end - start
	
	if height <= config.max_line_height:
		return [region]
	
	# Find the minimum projection value within this region
	region_proj = projection[start:end]
	min_val = region_proj.min()
	max_val = region_proj.max()
	
	# Use a threshold between min and the configured threshold
	split_threshold = min_val + (config.projection_threshold - min_val) * config.split_threshold_factor
	
	# Find the best split point (deepest valley)
	best_split = None
	best_depth = float('inf')
	
	for i, val in enumerate(region_proj):
		if val < split_threshold and val < best_depth:
			# Check it's not too close to edges
			if i > config.min_line_height and (len(region_proj) - i) > config.min_line_height:
				best_depth = val
				best_split = i
	
	if best_split is None:
		# No good split found, return as-is (will be flagged in metadata)
		return [region]
	
	# Split at the valley
	split_point = start + best_split
	left_region = (start, split_point)
	right_region = (split_point, end)
	
	# Recursively split if still too tall
	result = []
	result.extend(_split_tall_region(projection, left_region, config))
	result.extend(_split_tall_region(projection, right_region, config))
	
	return result


def _auto_split_regions(
	projection: np.ndarray,
	regions: List[Tuple[int, int]],
	config: LineSegmentationConfig,
) -> List[Tuple[int, int]]:
	"""Split any regions that exceed max_line_height."""
	if not config.enable_auto_split:
		return regions
	
	result = []
	for region in regions:
		result.extend(_split_tall_region(projection, region, config))
	
	# Filter out regions that are now too small
	result = [(s, e) for s, e in result if e - s >= config.min_line_height]
	
	return sorted(result, key=lambda r: r[0])


def _compute_ink_ratio(gray_crop: np.ndarray, threshold: int = 180) -> float:
	"""Compute the ratio of ink pixels in a grayscale crop."""
	_, binary = cv2.threshold(gray_crop, threshold, 255, cv2.THRESH_BINARY_INV)
	return np.count_nonzero(binary) / (binary.size + 1e-6)

#Pad a raw line span with configurable margins while clamping to page bounds.
def _expand_region(region: Tuple[int, int], height: int, config: LineSegmentationConfig) -> Tuple[int, int]:
	
	start, end = region
	return (
		max(0, start - config.pad_top),
		min(height, end + config.pad_bottom),
	)

#Resize a cropped line to the target height and add horizontal padding.
def _normalize_line_image(crop: np.ndarray, config: LineSegmentationConfig) -> np.ndarray:
	h, w = crop.shape[:2]
	if h == 0 or w == 0:
		return np.full((config.target_height, w or config.target_height), config.background_value, dtype=np.uint8)
	scale = config.target_height / h
	new_w = max(1, int(w * scale))
	resized = cv2.resize(crop, (new_w, config.target_height), interpolation=cv2.INTER_CUBIC)
	pad = config.horizontal_padding
	padded = cv2.copyMakeBorder(
		resized,
		0,
		0,
		pad,
		pad,
		cv2.BORDER_CONSTANT,
		value=config.background_value,
	)
	return padded


def _save_line_crop(
	line_img: np.ndarray,
	output_dir: Path,
	page_stem: str,
	index: int,
) -> Path:
	#Persist a normalized line image to disk and return its path.
	line_folder = output_dir / page_stem
	line_folder.mkdir(parents=True, exist_ok=True)
	line_path = line_folder / f"line_{index:03d}.png"
	cv2.imwrite(str(line_path), line_img)
	return line_path


def segment_lines_from_page(
	image_path: Path | str,
	output_dir: Path | str,
	config: LineSegmentationConfig | None = None,
	metadata_path: Path | None = None,
) -> dict:
	#Segment a single page into line crops and return the metadata payload.
	config = config or LineSegmentationConfig()
	image_path = Path(image_path)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	image = _load_image(image_path)
	gray = _ensure_grayscale(image)
	binary = _binarize(gray, config)
	projection = _projection_profile(binary, config)
	regions = _detect_line_regions(projection, config)
	
	# Auto-split oversized regions
	regions = _auto_split_regions(projection.ravel(), regions, config)
	
	height, _ = gray.shape
	lines_metadata = []
	skipped_lines = []
	page_stem = image_path.stem

	if config.max_lines is not None:
		regions = regions[: config.max_lines]

	line_index = 0
	for region in regions:
		expanded = _expand_region(region, height, config)
		crop = gray[expanded[0] : expanded[1], :]
		
		# Filter out low-ink regions (noise, empty areas)
		ink_ratio = _compute_ink_ratio(crop)
		if ink_ratio < config.min_ink_ratio:
			skipped_lines.append({
				"y_start": int(expanded[0]),
				"y_end": int(expanded[1]),
				"height": int(expanded[1] - expanded[0]),
				"reason": "low_ink",
				"ink_ratio": round(ink_ratio, 4),
			})
			continue
		
		line_index += 1
		normalized = _normalize_line_image(crop, config)
		line_path = _save_line_crop(normalized, output_dir, page_stem, line_index)
		
		# Flag if line is still oversized after splitting attempts
		line_height = expanded[1] - expanded[0]
		status = "ok" if line_height <= config.max_line_height else "oversized"
		
		lines_metadata.append(
			{
				"index": line_index,
				"y_start": int(expanded[0]),
				"y_end": int(expanded[1]),
				"height": line_height,
				"ink_ratio": round(ink_ratio, 4),
				"status": status,
				"path": str(line_path.resolve()),
			}
		)

	metadata = {
		"page": str(image_path.resolve()),
		"page_stem": page_stem,
		"config": asdict(config),
		"lines": lines_metadata,
		"skipped": skipped_lines,
		"stats": {
			"total_regions": len(regions),
			"exported_lines": len(lines_metadata),
			"skipped_low_ink": len(skipped_lines),
			"oversized_lines": sum(1 for l in lines_metadata if l.get("status") == "oversized"),
		},
	}
	if metadata_path:
		with open(metadata_path, "w", encoding="utf-8") as fh:
			json.dump(metadata, fh, indent=2)
	return metadata


def batch_segment_lines(
	pages_dir: Path | str,
	output_dir: Path | str,
	pattern: str = "*_preproc.png",
	config: LineSegmentationConfig | None = None,
	manifests_dir: Path | None = None,
) -> Iterator[dict]:
	#Run line segmentation over many pages, optionally saving manifests.
	pages_dir = Path(pages_dir)
	if not pages_dir.exists():
		raise FileNotFoundError(f"Pages directory missing: {pages_dir}")
	config = config or LineSegmentationConfig()
	output_dir = Path(output_dir)
	manifests_dir = Path(manifests_dir) if manifests_dir else None
	if manifests_dir:
		manifests_dir.mkdir(parents=True, exist_ok=True)

	for image_path in sorted(pages_dir.glob(pattern)):
		manifest_path = None
		if manifests_dir:
			manifest_path = manifests_dir / f"{image_path.stem}_lines.json"
		yield segment_lines_from_page(image_path, output_dir, config=config, metadata_path=manifest_path)


def _parse_args():
	"""Parse CLI options for batch line segmentation."""
	import argparse

	parser = argparse.ArgumentParser(description="Stage 2.1.3 line segmentation")
	parser.add_argument("pages_dir", type=Path, help="Directory with preprocessed pages (data/pages_preproc)")
	parser.add_argument("output_dir", type=Path, help="Destination root for line crops (data/lines)")
	parser.add_argument(
		"--pattern",
		default="*_preproc.png",
		help="Glob for selecting subset of preprocessed pages",
	)
	parser.add_argument(
		"--manifests-dir",
		type=Path,
		help="Optional directory to store per-page line metadata JSON files",
	)
	parser.add_argument("--projection-threshold", type=float, help="Normalized projection cutoff for line detection")
	parser.add_argument("--min-line-height", type=int, help="Minimum number of rows to treat as a line")
	parser.add_argument("--merge-gap", type=int, help="Merge line regions separated by <= this many rows")
	parser.add_argument("--pad-top", type=int, help="Top padding (pixels) added before cropping each line")
	parser.add_argument("--pad-bottom", type=int, help="Bottom padding (pixels) added after cropping each line")
	parser.add_argument("--target-height", type=int, help="Normalized line height in pixels")
	parser.add_argument("--horizontal-padding", type=int, help="Horizontal padding (pixels) added to both sides")
	parser.add_argument("--max-lines", type=int, help="Optional cap on number of exported lines per page")
	# New arguments for improved segmentation
	parser.add_argument("--max-line-height", type=int, help="Auto-split lines taller than this (default: 150)")
	parser.add_argument("--min-ink-ratio", type=float, help="Filter lines with ink ratio below this (default: 0.015)")
	parser.add_argument("--no-auto-split", action="store_true", help="Disable automatic splitting of oversized lines")
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = _parse_args()
	config_kwargs = {
		"projection_threshold": args.projection_threshold,
		"min_line_height": args.min_line_height,
		"merge_gap": args.merge_gap,
		"pad_top": args.pad_top,
		"pad_bottom": args.pad_bottom,
		"target_height": args.target_height,
		"horizontal_padding": args.horizontal_padding,
		"max_lines": args.max_lines,
		"max_line_height": args.max_line_height,
		"min_ink_ratio": args.min_ink_ratio,
		"enable_auto_split": not args.no_auto_split if hasattr(args, 'no_auto_split') else None,
	}
	config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}
	config = LineSegmentationConfig(**config_kwargs)
	for meta in batch_segment_lines(
		args.pages_dir,
		args.output_dir,
		pattern=args.pattern,
		config=config,
		manifests_dir=args.manifests_dir,
	):
		stats = meta.get('stats', {})
		print(f"[STAGE 2.1.3] {meta['page_stem']} -> {stats.get('exported_lines', len(meta['lines']))} lines "
		      f"(skipped: {stats.get('skipped_low_ink', 0)}, oversized: {stats.get('oversized_lines', 0)})")

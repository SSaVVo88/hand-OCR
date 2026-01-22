"""
Visualize projection profile and detected line regions for debugging.
Saves a diagnostic image showing where lines are being detected.
"""

import cv2
import numpy as np
import json
from pathlib import Path


def visualize_projection(page_path: Path, output_path: Path, config: dict):
    """Create visualization of projection profile and detected lines."""
    
    img = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    # Binary mask
    block = config.get('adaptive_block_size', 35)
    if block % 2 == 0:
        block += 1
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, config.get('adaptive_c', 10)
    )
    
    # Projection profile
    proj = binary.sum(axis=1).astype(np.float32)
    proj_norm = proj / (proj.max() + 1e-6)
    kernel = config.get('blur_kernel', 5)
    if kernel % 2 == 0:
        kernel += 1
    proj_smooth = cv2.GaussianBlur(proj_norm, (kernel, 1), 0).ravel()
    
    # Create visualization canvas
    proj_width = 300
    canvas = np.ones((h, w + proj_width), dtype=np.uint8) * 255
    canvas[:, :w] = img
    
    # Draw projection profile
    for y in range(h):
        bar_len = int(proj_smooth[y] * (proj_width - 20))
        if bar_len > 0:
            canvas[y, w+10:w+10+bar_len] = 180
    
    # Draw threshold line
    threshold = config.get('projection_threshold', 0.12)
    thresh_x = w + 10 + int(threshold * (proj_width - 20))
    cv2.line(canvas, (thresh_x, 0), (thresh_x, h), 100, 2)
    
    # Detect and draw line regions
    min_height = config.get('min_line_height', 32)
    merge_gap = config.get('merge_gap', 18)
    
    regions = []
    in_line = False
    start = 0
    
    for idx, val in enumerate(proj_smooth):
        if not in_line and val >= threshold:
            in_line = True
            start = idx
        elif in_line and val < threshold:
            if idx - start >= min_height:
                regions.append((start, idx))
            in_line = False
    if in_line:
        regions.append((start, h))
    
    # Merge regions
    merged = []
    for start, end in regions:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    
    # Draw detected regions on the image
    for i, (y1, y2) in enumerate(merged):
        height = y2 - y1
        if height > 150:
            color = (0, 0, 255)  # Red - merged
        elif height < 50:
            color = (0, 165, 255)  # Orange - small
        else:
            color = (0, 255, 0)  # Green - OK
        
        # Draw on color version
        cv2.rectangle(canvas, (0, y1), (w-1, y2), 128, 2)
        cv2.putText(canvas, f"L{i+1}:{height}px", (10, y1+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    
    cv2.imwrite(str(output_path), canvas)
    print(f"Saved: {output_path}")
    print(f"  Detected {len(merged)} regions")
    
    return merged


def main():
    project_root = Path(__file__).resolve().parents[1]
    pages_preproc = project_root / "data" / "pages_preproc"
    output_dir = project_root / "experiments" / "debug_segmentation"
    output_dir.mkdir(exist_ok=True)
    
    # Current config
    config = {
        'blur_kernel': 5,
        'adaptive_block_size': 35,
        'adaptive_c': 10,
        'projection_threshold': 0.12,
        'min_line_height': 32,
        'merge_gap': 18,
    }
    
    # Analyze skan_001
    page = pages_preproc / "skan_001_page_001_preproc.png"
    if page.exists():
        visualize_projection(page, output_dir / "skan_001_projection.png", config)
        
        # Try with adjusted params
        print("\n--- Testing with adjusted params ---")
        config_v2 = {
            'blur_kernel': 3,  # Less smoothing
            'adaptive_block_size': 25,  # Smaller blocks
            'adaptive_c': 12,
            'projection_threshold': 0.08,  # Lower threshold to catch gaps
            'min_line_height': 40,  # Higher minimum
            'merge_gap': 8,  # Less aggressive merging
        }
        visualize_projection(page, output_dir / "skan_001_projection_v2.png", config_v2)


if __name__ == "__main__":
    main()

"""
Deep analysis of projection profile to find optimal segmentation parameters.
"""

import cv2
import numpy as np
from pathlib import Path


def analyze_gaps(page_path: str):
    """Find natural gaps in handwriting to understand line spacing."""
    
    img = cv2.imread(page_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    print(f"Page: {h}x{w}")
    
    # Binary mask
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 35, 10
    )
    
    # Raw projection (no smoothing)
    proj = binary.sum(axis=1).astype(np.float32)
    proj_norm = proj / (proj.max() + 1e-6)
    
    # Find gap runs at different thresholds
    for gap_threshold in [0.02, 0.05, 0.08, 0.10]:
        in_gap = False
        gap_runs = []
        start = 0
        
        for i, val in enumerate(proj_norm):
            if not in_gap and val < gap_threshold:
                in_gap = True
                start = i
            elif in_gap and val >= gap_threshold:
                gap_runs.append((start, i, i - start))
                in_gap = False
        
        # Filter significant gaps (>5px)
        significant_gaps = [g for g in gap_runs if g[2] > 5]
        
        print(f"\n  threshold={gap_threshold}: {len(significant_gaps)} significant gaps")
        if significant_gaps:
            gap_sizes = [g[2] for g in significant_gaps]
            print(f"    Gap sizes: min={min(gap_sizes)}, max={max(gap_sizes)}, median={np.median(gap_sizes):.0f}")


def test_segmentation_params(page_path: str, configs: list):
    """Test different segmentation parameters."""
    
    img = cv2.imread(page_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    for name, cfg in configs:
        block = cfg['adaptive_block_size']
        if block % 2 == 0:
            block += 1
            
        binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block, cfg['adaptive_c']
        )
        
        proj = binary.sum(axis=1).astype(np.float32)
        proj_norm = proj / (proj.max() + 1e-6)
        
        kernel = cfg['blur_kernel']
        if kernel % 2 == 0:
            kernel += 1
        proj_smooth = cv2.GaussianBlur(proj_norm, (kernel, 1), 0).ravel()
        
        # Detect regions
        threshold = cfg['projection_threshold']
        min_height = cfg['min_line_height']
        merge_gap = cfg['merge_gap']
        
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
        if in_line and h - start >= min_height:
            regions.append((start, h))
        
        # Merge
        merged = []
        for s, e in regions:
            if merged and s - merged[-1][1] <= merge_gap:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        
        heights = [e - s for s, e in merged]
        
        print(f"\n{name}:")
        print(f"  Detected: {len(merged)} lines")
        if heights:
            print(f"  Heights: {heights}")
            print(f"  Range: {min(heights)}-{max(heights)}px, median: {np.median(heights):.0f}px")
            ok_lines = sum(1 for h in heights if 50 <= h <= 120)
            print(f"  OK lines (50-120px): {ok_lines}/{len(heights)}")


def main():
    page = "data/pages_preproc/skan_001_page_001_preproc.png"
    
    print("=" * 60)
    print("GAP ANALYSIS")
    print("=" * 60)
    analyze_gaps(page)
    
    print("\n" + "=" * 60)
    print("PARAMETER TESTING")
    print("=" * 60)
    
    configs = [
        ("CURRENT (default)", {
            'blur_kernel': 5,
            'adaptive_block_size': 35,
            'adaptive_c': 10,
            'projection_threshold': 0.12,
            'min_line_height': 32,
            'merge_gap': 18,
        }),
        ("OPTION A: Lower merge_gap", {
            'blur_kernel': 5,
            'adaptive_block_size': 35,
            'adaptive_c': 10,
            'projection_threshold': 0.12,
            'min_line_height': 32,
            'merge_gap': 5,  # Much less merging
        }),
        ("OPTION B: Lower threshold + less merge", {
            'blur_kernel': 5,
            'adaptive_block_size': 35,
            'adaptive_c': 10,
            'projection_threshold': 0.08,  # More sensitive
            'min_line_height': 40,
            'merge_gap': 8,
        }),
        ("OPTION C: Less blur + lower threshold", {
            'blur_kernel': 3,
            'adaptive_block_size': 35,
            'adaptive_c': 10,
            'projection_threshold': 0.06,
            'min_line_height': 35,
            'merge_gap': 5,
        }),
        ("OPTION D: Smaller block + lower threshold", {
            'blur_kernel': 3,
            'adaptive_block_size': 25,
            'adaptive_c': 8,
            'projection_threshold': 0.05,
            'min_line_height': 35,
            'merge_gap': 4,
        }),
    ]
    
    test_segmentation_params(page, configs)
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("""
The main issue is that the current parameters merge adjacent lines.

Root causes:
1. projection_threshold=0.12 is too high - gaps between closely spaced 
   lines still have projection > 0.12 due to ascenders/descenders
2. merge_gap=18 aggressively merges regions that are close
3. The blur_kernel=5 smooths out fine gaps between lines

Key insight: Your handwriting has tight line spacing, so gaps between
lines have some ink from ascenders (g, y, j) and descenders (b, d, f).
The projection never drops below 0.12 in these gaps.

Best approach: Lower threshold + minimal merge_gap to catch small gaps.
""")


if __name__ == "__main__":
    main()

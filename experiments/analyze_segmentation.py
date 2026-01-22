"""
Diagnostic script to analyze line segmentation quality and projection profiles.
"""

import cv2
import numpy as np
import json
from pathlib import Path

def analyze_page(page_path: Path, manifest_path: Path):
    """Analyze a single page's segmentation quality."""
    
    # Load page
    img = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
    print(f"\n=== Analyzing: {page_path.name} ===")
    print(f"Page dimensions: {img.shape}")
    
    # Create binary with segmentation params
    block = 35
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, block, 10
    )
    
    # Compute projection
    proj = binary.sum(axis=1).astype(np.float32)
    proj_norm = proj / (proj.max() + 1e-6)
    proj_smooth = cv2.GaussianBlur(proj_norm, (5, 1), 0).ravel()
    
    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    print(f"\nDetected {len(manifest['lines'])} lines")
    print("\nLine analysis:")
    
    for line in manifest['lines']:
        h = line['height']
        # Classify line quality
        if h < 60:
            status = "⚠️  SMALL (possibly noise/partial)"
        elif h > 150:
            status = "❌ LARGE (possibly merged lines)"
        else:
            status = "✅ OK"
        
        print(f"  line_{line['index']:03d}: {h:3d}px  {status}")
    
    # Estimate expected line height
    heights = [l['height'] for l in manifest['lines']]
    small_lines = [h for h in heights if 50 <= h <= 120]
    
    if small_lines:
        expected_height = np.median(small_lines)
        print(f"\nEstimated single-line height: ~{expected_height:.0f}px")
        print(f"Lines likely merged: {sum(1 for h in heights if h > expected_height * 1.8)}")
        print(f"Lines likely noise: {sum(1 for h in heights if h < 50)}")
    
    return heights


def main():
    project_root = Path(__file__).resolve().parents[1]
    pages_preproc = project_root / "data" / "pages_preproc"
    manifests_dir = project_root / "data" / "lines_manifests"
    
    # Analyze a few sample pages
    samples = [
        "skan_001_page_001_preproc",
        "skan_002_page_001_preproc", 
        "skan_010_page_001_preproc",
    ]
    
    all_heights = []
    
    for stem in samples:
        page_path = pages_preproc / f"{stem}.png"
        manifest_path = manifests_dir / f"{stem}_lines.json"
        
        if page_path.exists() and manifest_path.exists():
            heights = analyze_page(page_path, manifest_path)
            all_heights.extend(heights)
    
    print("\n" + "=" * 50)
    print("SUMMARY ACROSS ALL ANALYZED PAGES")
    print("=" * 50)
    print(f"Total lines: {len(all_heights)}")
    print(f"Height range: {min(all_heights)}-{max(all_heights)}px")
    print(f"Mean: {np.mean(all_heights):.1f}px, Median: {np.median(all_heights):.1f}px")
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    print("1. Lines >150px are likely merged - consider lowering merge_gap or projection_threshold")
    print("2. Lines <50px may be noise - consider raising min_line_height")
    print("3. Current projection_threshold=0.12 may be too low for dense handwriting")


if __name__ == "__main__":
    main()

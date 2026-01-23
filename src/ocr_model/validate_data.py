"""
Validate labeled data before training.
"""
import csv
from pathlib import Path
from collections import Counter

def validate_data():
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / 'data' / 'lines.csv'
    lines_dir = project_root / 'data' / 'lines'
    
    found = []
    missing = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            img_name = row['image_name']
            transcription = row['transcription'].strip()
            
            # Parse names like "skan_026_page_001_preproc_line_001.png"
            # to folder "skan_026_page_001_preproc" and file "line_001.png"
            base_name = img_name.replace('.png', '')
            
            # Find "_line_" separator to split folder from filename
            if '_line_' not in base_name:
                missing.append((img_name, "bad format - no _line_ separator"))
                continue
            
            parts = base_name.rsplit('_line_', 1)
            folder = parts[0]
            line_num = parts[1]
            img_path = lines_dir / folder / f'line_{line_num}.png'
            
            if img_path.exists():
                found.append({
                    'path': str(img_path),
                    'text': transcription,
                    'length': len(transcription)
                })
            else:
                missing.append((img_name, "file not found"))
    
    print("=" * 60)
    print("Data Validation Report")
    print("=" * 60)
    print(f"\n✓ Found: {len(found)} valid labeled samples")
    print(f"✗ Missing: {len(missing)} samples")
    
    if found:
        lengths = [f['length'] for f in found]
        print(f"\nText length stats:")
        print(f"  Min: {min(lengths)} chars")
        print(f"  Max: {max(lengths)} chars")
        print(f"  Avg: {sum(lengths)/len(lengths):.1f} chars")
        
        print(f"\nSample entries:")
        for item in found[:5]:
            print(f"  {Path(item['path']).name}: \"{item['text'][:50]}...\"")
    
    if missing:
        print(f"\nMissing samples (first 10):")
        for name, reason in missing[:10]:
            print(f"  {name}: {reason}")
    
    return found, missing

if __name__ == "__main__":
    validate_data()

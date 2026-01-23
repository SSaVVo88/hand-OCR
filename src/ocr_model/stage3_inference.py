"""
TrOCR Inference Script

Test the trained model or the base model on handwriting images.
"""

import argparse
from pathlib import Path
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def load_model(model_path: str = None):
    """Load TrOCR model and processor."""
    if model_path and Path(model_path).exists():
        print(f"Loading fine-tuned model from: {model_path}")
        processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
    else:
        print("Loading base TrOCR model (microsoft/trocr-small-handwritten)")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
    
    # Use MPS on Mac if available, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    model.to(device)
    model.eval()
    
    return processor, model, device


def recognize_image(image_path: str, processor, model, device) -> str:
    """Recognize text from a single image."""
    image = Image.open(image_path).convert("RGB")
    
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=128)
    
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


def main():
    parser = argparse.ArgumentParser(description="TrOCR Inference")
    parser.add_argument("image", type=str, nargs="?", help="Path to image file")
    parser.add_argument("--model", type=str, default=None, help="Path to fine-tuned model")
    parser.add_argument("--batch", type=str, help="Process all images in directory")
    args = parser.parse_args()
    
    # Load model
    processor, model, device = load_model(args.model)
    
    if args.batch:
        # Process directory
        batch_dir = Path(args.batch)
        images = list(batch_dir.glob("*.png")) + list(batch_dir.glob("*.jpg"))
        print(f"\nProcessing {len(images)} images from {batch_dir}")
        
        for img_path in sorted(images)[:20]:  # Limit to 20 for demo
            text = recognize_image(str(img_path), processor, model, device)
            print(f"{img_path.name}: {text}")
    
    elif args.image:
        # Process single image
        text = recognize_image(args.image, processor, model, device)
        print(f"\nRecognized text: {text}")
    
    else:
        # Demo mode - process a few samples
        project_root = Path(__file__).resolve().parents[2]
        lines_dir = project_root / "data" / "lines"
        
        # Find some sample images
        sample_dirs = sorted(lines_dir.iterdir())[:3]
        
        print("\n" + "=" * 60)
        print("Demo: Testing TrOCR on sample images")
        print("=" * 60)
        
        for sample_dir in sample_dirs:
            if not sample_dir.is_dir():
                continue
            images = sorted(sample_dir.glob("*.png"))[:3]
            
            print(f"\n--- {sample_dir.name} ---")
            for img_path in images:
                text = recognize_image(str(img_path), processor, model, device)
                print(f"  {img_path.name}: {text}")


if __name__ == "__main__":
    main()

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from PIL import Image
import io
from pathlib import Path
import torch

# OCR Model imports
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# Żeby uruchomić serwer wpisujemy w katalogu głównym (hand-OCR):
# uvicorn src.app.API:app --host 0.0.0.0 --port 8000
# Następnie otwieramy plik 'index.html' normalnie w przeglądarce


# Tworzenie aplikacji
app = FastAPI()

# Zabezpieczenia CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# Global model storage
ocr_model = None
ocr_processor = None
device = None


def load_ocr_model():
    """Load OCR model on startup."""
    global ocr_model, ocr_processor, device
    
    # Check for fine-tuned model first
    project_root = Path(__file__).resolve().parents[2]
    finetuned_path = project_root / "models" / "trocr-polish-handwriting" / "final"
    
    if finetuned_path.exists():
        print(f"Loading fine-tuned model from: {finetuned_path}")
        ocr_processor = TrOCRProcessor.from_pretrained(str(finetuned_path))
        ocr_model = VisionEncoderDecoderModel.from_pretrained(str(finetuned_path))
    else:
        print("Fine-tuned model not found, using base TrOCR model")
        model_name = "microsoft/trocr-small-handwritten"
        ocr_processor = TrOCRProcessor.from_pretrained(model_name)
        ocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    ocr_model.to(device)
    ocr_model.eval()
    print(f"OCR model loaded on {device}")


@app.on_event("startup")
async def startup_event():
    """Load model when server starts."""
    load_ocr_model()


# Ogólny handler błędów
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "Unexpected error occurred."
        },
    )


# handler /predict
@app.post("/predict")
async def predict(file: UploadFile = File()):
    # Sprawdzanie typu pliku
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="The file must be an image.")

    # Odczytanie pliku
    try:
        content = await file.read()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to read the file.")

    # Sprawdzanie rozmiaru
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Uploaded file is too big (max 5MB).")

    # sprawdzanie pliku
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()  # sprawdzenie integralności
        img = Image.open(io.BytesIO(content))  # ponowne otwarcie
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Corrupted file.")

    width, height = img.size
    
    # Run OCR inference
    recognized_text = ""
    if ocr_model is not None and ocr_processor is not None:
        try:
            # Convert to RGB
            img_rgb = img.convert("RGB")
            
            # Process image
            pixel_values = ocr_processor(img_rgb, return_tensors="pt").pixel_values.to(device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = ocr_model.generate(pixel_values, max_length=128)
            
            recognized_text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception as e:
            recognized_text = f"OCR error: {str(e)}"
    else:
        recognized_text = "Model not loaded"
    
    return {
        "filename": file.filename,
        "size": {
            "width": width,
            "height": height
        },
        "text": recognized_text
    }


@app.get("/health")
async def health():
    model_status = "loaded" if ocr_model is not None else "not loaded"
    return {"status": "ok", "model": model_status}
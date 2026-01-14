from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Poniższe tylko do otwierania pliku png (możliwe że w przyszłości niepotrzebne)
from PIL import Image
import io


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
    return {
        "filename": file.filename,
        "size": {
            "width": width,
            "height": height}}


@app.get("/health")
async def health():
    return {"status": "ok"}
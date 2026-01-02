from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

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


# handler /predict
@app.post("/predict")
async def predict(file: UploadFile = File(), author: str = Form()):
    # Sprawdzanie typu pliku
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # Odczytanie pliku - w celach testowych
    img = await file.read()
    try:
        img = Image.open(io.BytesIO(img))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    width, height = img.size
    imiona = {"Zuzanna":"Zuzanna Heldt", "Konrad":"Konrad Hennig", "Emilia":"Emilia Kreft", "Piotr":"Piotr Przypaśniak", "Przemek":"Przemysław Sawoniuk"}
    return {
        "filename": file.filename,
        "size": {
            "width": width,
            "height": height},
        "author": imiona[author]}

# Sprawdzanie statusu
@app.get("/health")
async def health():
    return {"status": "ok"}
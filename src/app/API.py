from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import webbrowser
import threading
import time

# Poniższe tylko do otwierania pliku png (możliwe że w przyszłości niepotrzebne)
from PIL import Image
import io


# Żeby uruchomić serwer wpisujemy w katalogu głównym (hand-OCR):
# uvicorn src.app.API:app


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
async def predict(file: UploadFile = File(...)):
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
    return {
        "filename": file.filename,
        "size": {
            "width": width,
            "height": height}}


# Automatyczne uruchamianie index.html
@app.get("/")
def serve_index():
    return FileResponse("src/app/index.html")

@app.get("/scripts.js")
def serve_js():
    return FileResponse("src/app/scripts.js")

def open_browser():
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:8000")

@app.on_event("startup")
def startup_event():
    threading.Thread(target=open_browser).start()

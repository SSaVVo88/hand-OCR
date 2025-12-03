from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import webbrowser
import threading
import time


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


# Klasa dla /predict
class PredictRequest(BaseModel):
    text: str


# handler /predict
@app.post("/predict")
def predict(request: PredictRequest):
    # W przyszłości tutaj prawdziwy predict z modelu
    reversed_text = request.text[::-1]
    return {"text_out": reversed_text}


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

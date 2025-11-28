from transformers import TrOCRProcessor, TrOCRForCausalLM
from PIL import Image
import torch

#Chcialem sprawdzic czy na Macbooku z M4 zadziala TrOCR z MPS (Metal Performance Shaders)

# Sprawdź, czy M4 używa MPS (Metal)
print("MPS available:", torch.backends.mps.is_available())

image = Image.open("ocr-test/sample_sentence.png").convert("RGB")

# Inicjalizuj model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
model = TrOCRForCausalLM.from_pretrained("microsoft/trocr-small-printed")

# Włącz MPS (dla M4)

device =  "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

inputs = processor(image, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(device)

# Generuj tekst
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Rozpoznany tekst:", text)
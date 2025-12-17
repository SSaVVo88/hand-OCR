from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image
import torch

# Sprawdź, czy M4 używa MPS (Metal)
print("MPS available:", torch.backends.mps.is_available())


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")

#device = "mps" if torch.backends.mps.is_available() else "cpu"
#model.to(device)

# load image from the IAM dataset
#url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg" (problemy z ceryfikatem SSL)
image = Image.open("sample.jpg").convert("RGB")
#image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Rozpoznany tekst:", generated_text)
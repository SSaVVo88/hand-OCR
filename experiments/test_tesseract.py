from PIL import Image, ImageDraw, ImageFont
# W tym skyrpcie tworzymy przykładowy obraz z tekstem w języku polskim na szybko
# zeby sprawdzic czy zadziala Tesseract OCR z jezykiem polskim.
# tesseract instalujemy poprzez pip install pytesseract tesseract-lang (zeby byl jezyk polski)
# albo brew install tesseract tesseract-lang(MacOS)


# Stwórz biały obraz 500x200
img = Image.new("RGB", (500, 200), color="white")
d = ImageDraw.Draw(img)

# Napisz tekst w języku polskim
d.text((50, 50), "Przykładowy tekst w języku polskim\nOCR test 123", fill="black")

# Zapisz
img.save("sample.jpg")


### Użyj poniższego polecenia w terminalu, aby uruchomić Tesseract OCR:
#  tesseract sample.jpg output -l eng+pol --psm 6

# Powstanie plik output.txt z rozpoznanym tekstem z obrazu.

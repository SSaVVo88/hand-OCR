# Handwriting OCR Model
## 🚀 Quick Start  
```bash
git clone https://github.com/SSaVVo88/hand-OCR.git
cd hand-OCR
```
## Create and activate virutal enviroment 
### Mac/Linux
```bash
python -m venv .venv
source .venv/bin/activate
```
### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```
### Install the project 
```bash
pip install -e .  # Installs YOUR CODE + dependencies
```
### Verify installation
```bash
pip list | grep ocr-model  # Should show: ocr-model 0.1.0 (editable)
```
### Run the model
```bash
python -m ocr_model.train
```

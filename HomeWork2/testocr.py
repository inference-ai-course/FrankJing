from PIL import Image
import pytesseract 
import os
import logging

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

script_dir = os.path.dirname(os.path.abspath(__file__))  # Folder of this script
image_path = os.path.join(script_dir, "img", "test.png")
image = Image.open(image_path)

#perform OCR on the image
text = pytesseract.image_to_string(image)

file_name = "image_to_text.txt"
file_path = os.path.join(script_dir,file_name)
with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)


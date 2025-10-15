import os
import shutil
from classifier import classify_image

# Carpeta original y de destino
FOLDER_PATH = "all"
OUTPUT_FOLDER = "all_renamed"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Contadores por clase
counters = {}

for filename in os.listdir(FOLDER_PATH):
    file_path = os.path.join(FOLDER_PATH, filename)
    ext = os.path.splitext(filename)[1].lower()

    if os.path.isfile(file_path) and ext in IMAGE_EXTENSIONS:
        class_name, confidence = classify_image(file_path)

        if class_name not in counters:
            counters[class_name] = 0

        new_name = f"{class_name}_{counters[class_name]}{ext}"
        new_path = os.path.join(OUTPUT_FOLDER, new_name)

        shutil.copy(file_path, new_path)
        counters[class_name] += 1

        print(f"{filename}: Clase = {class_name}, Confianza = {confidence:.2f} -> {new_name}")

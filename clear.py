import os
import shutil

# Carpetas a eliminar
folders_to_delete = ["all", "all_renamed", "cats", "dogs", "mice", "rejected_mice"]

for folder in folders_to_delete:
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Carpeta eliminada: {folder}")
    else:
        print(f"La carpeta no existe: {folder}")
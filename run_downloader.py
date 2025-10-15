import os
import asyncio
import aiohttp
from urllib.parse import urlparse
from tqdm import tqdm
import hashlib
import random
import io
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from dotenv import load_dotenv
load_dotenv()

# === CONFIGURACIÓN ===
DOG_API = "https://dog.ceo/api/breeds/image/random"
CAT_API = "https://api.thecatapi.com/v1/images/search"

PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")
MOUSE_API = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q=mouse+animal&image_type=photo&per_page=200"

DOG_DIR = os.path.join("dogs")
CAT_DIR = os.path.join("cats")
MOUSE_DIR = os.path.join("mice")
REJECTED_DIR = os.path.join("rejected_mice")
ALL_DIR = os.path.join("all")

for folder in [DOG_DIR, CAT_DIR, MOUSE_DIR, REJECTED_DIR, ALL_DIR]:
    os.makedirs(folder, exist_ok=True)

SEM_LIMIT = 25

# === CARGAR MODELO LOCAL ===
print("🧠 Cargando modelo OWL-ViT (detección local de ratones)...")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
model.eval()
print("✅ Modelo cargado correctamente.\n")


# === FUNCIÓN DE VERIFICACIÓN LOCAL ===
def verify_mouse_image_local(image_bytes, threshold=0.25):
    """Verifica si hay un ratón o roedor en la imagen usando OWL-ViT localmente."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        text_labels = [["a photo of a mouse", "a rodent", "a rat"]]

        inputs = processor(text=text_labels, images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([(image.height, image.width)])
        results = processor.post_process_grounded_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes.tolist(),
            text_labels=text_labels
        )

        res = results[0]
        scores = res["scores"]
        labels = res["labels"]

        for score, label in zip(scores, labels):
            lbl = text_labels[0][label]
            if any(k in lbl for k in ["mouse", "rodent", "rat"]):
                return True
        return False

    except Exception as e:
        print(f"⚠️ Error en verificación local: {e}")
        return True  # aceptar por defecto si hay error


# === DESCARGA DE IMAGENES ===
async def download_image(session, url, folder, sem, verify=False):
    async with sem:
        try:
            async with session.get(url, timeout=15) as resp:
                if resp.status != 200:
                    return False
                content = await resp.read()

                # Guardar en carpeta "all"
                filename_all = hashlib.sha1(content).hexdigest() + os.path.splitext(urlparse(url).path)[-1]
                filepath_all = os.path.join(ALL_DIR, filename_all)
                if not os.path.exists(filepath_all):
                    with open(filepath_all, "wb") as f:
                        f.write(content)

                # Verificación local si aplica
                if verify:
                    loop = asyncio.get_event_loop()
                    is_valid = await loop.run_in_executor(None, verify_mouse_image_local, content)
                    if not is_valid:
                        filename = hashlib.sha1(content).hexdigest() + os.path.splitext(urlparse(url).path)[-1]
                        filepath = os.path.join(REJECTED_DIR, filename)
                        with open(filepath, "wb") as f:
                            f.write(content)
                        return False

                filename = hashlib.sha1(content).hexdigest() + os.path.splitext(urlparse(url).path)[-1]
                filepath = os.path.join(folder, filename)
                if os.path.exists(filepath):
                    return False

                with open(filepath, "wb") as f:
                    f.write(content)
                return True
        except Exception as e:
            print(f"⚠️ Error al descargar/verificar {url}: {e}")
            return False


# === FUNCIONES PARA OBTENER URLS ===
async def fetch_dog_url(session):
    async with session.get(DOG_API) as resp:
        data = await resp.json()
        return data["message"]


async def fetch_cat_url(session):
    async with session.get(CAT_API) as resp:
        data = await resp.json()
        return data[0]["url"]


async def get_mouse_urls(session):
    """Descarga una lista de URLs de ratones desde Pixabay."""
    async with session.get(MOUSE_API) as resp:
        data = await resp.json()
        return [img["largeImageURL"] for img in data.get("hits", [])]


# === DESCARGA POR CATEGORÍA ===
async def download_category(session, fetch_fn, folder, count, pbar, urls_source=None, verify=False):
    sem = asyncio.Semaphore(SEM_LIMIT)
    urls = set()
    tasks = []

    urls_needed = count * 2 if verify else count

    while len(urls) < urls_needed:
        if urls_source:
            url = random.choice(urls_source)
        else:
            url = await fetch_fn(session)
        if url not in urls:
            urls.add(url)
            tasks.append(download_image(session, url, folder, sem, verify=verify))

    successful = 0
    for task in asyncio.as_completed(tasks):
        success = await task
        if success:
            successful += 1
            pbar.update(1)
        if successful >= count:
            break


# === MAIN ===
async def main():
    async with aiohttp.ClientSession() as session:
        mouse_urls = await get_mouse_urls(session)
        if not mouse_urls:
            print("⚠️ No se pudieron obtener imágenes de ratones desde Pixabay.")
            return

        print("🤖 Verificando localmente imágenes de ratones con OWL-ViT...")
        print("   (Las imágenes rechazadas se guardarán en 'images/rejected_mice')\n")

        total_per_category = 100

        pbar_dogs = tqdm(total=total_per_category, desc="🐶 Perros ", position=0, leave=True, unit="img",
                         ncols=80, dynamic_ncols=False, colour='blue')
        pbar_cats = tqdm(total=total_per_category, desc="🐱 Gatos  ", position=1, leave=True, unit="img",
                         ncols=80, dynamic_ncols=False, colour='green')
        pbar_mice = tqdm(total=total_per_category, desc="🐭 Ratones", position=2, leave=True, unit="img",
                         ncols=80, dynamic_ncols=False, colour='yellow')

        await asyncio.gather(
            download_category(session, fetch_dog_url, DOG_DIR, total_per_category, pbar_dogs),
            download_category(session, fetch_cat_url, CAT_DIR, total_per_category, pbar_cats),
            download_category(session, None, MOUSE_DIR, total_per_category, pbar_mice, urls_source=mouse_urls, verify=True),
        )

        for bar in [pbar_dogs, pbar_cats, pbar_mice]:
            bar.close()

        print("\n✅ Descargas completadas con éxito.")
        print(f"   Las imágenes rechazadas están en: {REJECTED_DIR}")


if __name__ == "__main__":
    asyncio.run(main())

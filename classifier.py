from keras.layers import TFSMLayer
from PIL import Image, ImageOps
import numpy as np
import os
import tensorflow as tf

# Ruta del SavedModel de TensorFlow
MODEL_PATH = os.path.join("model", "model.savedmodel")
LABELS_PATH = os.path.join("model", "labels.txt")

# Cargar etiquetas
with open(LABELS_PATH, "r") as f:
    class_names = [line.strip().lower() for line in f.readlines()]

# Crear una capa que use el SavedModel para inferencia
model = TFSMLayer(MODEL_PATH, call_endpoint="serving_default")


def classify_image(image_path: str, confidence_threshold: float = 0.5):
    """
    Clasifica una imagen usando el SavedModel en Keras 3.
    Devuelve 'ninguno' si la confianza es menor al threshold.

    Args:
        image_path (str): Ruta de la imagen.
        confidence_threshold (float): Umbral mínimo de confianza para considerar la predicción válida.

    Returns:
        tuple: (nombre_clase, confianza)
    """
    # Cargar y procesar imagen
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (image_array / 127.5) - 1
    data = tf.convert_to_tensor([normalized_image_array])  # tensor 4D

    # Predecir usando TFSMLayer
    prediction = model.call(data)  # devuelve un diccionario {endpoint_name: tensor}

    # Extraer el tensor de la salida
    output_tensor = list(prediction.values())[0]
    prediction_array = output_tensor.numpy()  # convertir a numpy

    # Obtener clase con mayor probabilidad
    index = np.argmax(prediction_array)
    confidence_score = float(prediction_array[0][index])
    class_name = class_names[index]

    # Devolver "ninguno" si la confianza es baja
    if confidence_score < confidence_threshold:
        class_name = "ninguno"

    return class_name, confidence_score

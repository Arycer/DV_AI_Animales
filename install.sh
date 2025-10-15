#!/bin/bash

# Script de instalación para DV_AI_Animales
# Este script instala todos los paquetes de Python requeridos para ejecutar el proyecto

echo "🚀 Iniciando instalación de dependencias para DV_AI_Animales..."

# Verificar si Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 no está instalado. Por favor, instala Python 3 y vuelve a ejecutar este script."
    exit 1
fi

echo "✅ Python 3 detectado"

# Crear entorno virtual si no existe
if [ ! -d ".venv" ]; then
    echo "🔧 Creando entorno virtual..."
    python3 -m venv .venv
    echo "✅ Entorno virtual creado"
else
    echo "✅ Entorno virtual ya existe"
fi

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source .venv/bin/activate

# Actualizar pip
echo "🔧 Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo "🔧 Instalando dependencias desde requirements.txt..."
pip install -r requirements.txt

echo "✅ Instalación completada con éxito!"
echo "🔍 Para activar el entorno virtual en el futuro, ejecuta: source .venv/bin/activate"
echo "🚀 Ahora puedes ejecutar los scripts del proyecto"
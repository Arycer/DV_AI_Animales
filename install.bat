@echo off
REM Script de instalación para DV_AI_Animales (Windows)
REM Este script instala todos los paquetes de Python requeridos para ejecutar el proyecto

echo 🚀 Iniciando instalación de dependencias para DV_AI_Animales...

REM Verificar si Python está instalado
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Python no está instalado. Por favor, instala Python y vuelve a ejecutar este script.
    exit /b 1
)

echo ✅ Python detectado

REM Crear entorno virtual si no existe
if not exist .venv\ (
    echo 🔧 Creando entorno virtual...
    python -m venv .venv
    echo ✅ Entorno virtual creado
) else (
    echo ✅ Entorno virtual ya existe
)

REM Activar entorno virtual
echo 🔧 Activando entorno virtual...
call .venv\Scripts\activate.bat

REM Actualizar pip
echo 🔧 Actualizando pip...
python -m pip install --upgrade pip

REM Instalar dependencias
echo 🔧 Instalando dependencias desde requirements.txt...
pip install -r requirements.txt

echo ✅ Instalación completada con éxito!
echo 🔍 Para activar el entorno virtual en el futuro, ejecuta: .venv\Scripts\activate.bat
echo 🚀 Ahora puedes ejecutar los scripts del proyecto

pause
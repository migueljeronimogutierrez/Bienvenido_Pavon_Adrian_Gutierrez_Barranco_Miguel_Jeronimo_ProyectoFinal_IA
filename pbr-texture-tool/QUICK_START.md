# English

# Quick Start for Local Evaluation

## Recommended steps

1. Extract the project ZIP to a local folder.
2. Make sure the model files are already inside:
   - `models/deeppbr/`
   - `models/cyclegan/`
3. Double-click `setup_env.bat` and wait until installation finishes.
4. Double-click `run_app.bat`.
5. A local Streamlit address should open in the browser.
6. Upload an image and run the pipeline.

## Important notes

- Recommended operating system: **Windows 10/11**
- Recommended Python version: **Python 3.12.x**
- CPU execution is supported
- The first run may take longer because dependencies must be installed
- Very high-resolution images may take more time to process

## Minimal folder check

Expected key files and folders:

```text
app/
models/deeppbr/
models/cyclegan/
app.py
requirements.txt
setup_env.bat
run_app.bat
```

## If the app does not start

Check the following:

- Python 3.12 is installed
- The model files are placed in the correct folders
- `setup_env.bat` finished without errors
- The folder path does not contain unusual permission restrictions

---

# Español

# Inicio rápido para evaluación local

## Pasos recomendados

1. Extrae el ZIP del proyecto en una carpeta local.
2. Comprueba que los archivos de modelos ya están dentro:
   - `models/deeppbr/`
   - `models/cyclegan/`
3. Haz doble clic en `setup_env.bat` y espera a que la instalación termine.
4. Haz doble clic en `run_app.bat`.
5. Se debería abrir una dirección local de Streamlit en el navegador.
6. Sube una imagen y ejecuta el pipeline.

## Notas importantes

- Sistema operativo recomendado: **Windows 10/11**
- Versión de Python recomendada: **Python 3.12.x**
- La ejecución en CPU está soportada
- La primera ejecución puede tardar más porque las dependencias deben instalarse
- Las imágenes de resolución muy alta pueden requerir más tiempo de procesamiento

## Comprobación mínima de carpetas

Archivos y carpetas clave esperados:

```text
app/
models/deeppbr/
models/cyclegan/
app.py
requirements.txt
setup_env.bat
run_app.bat
```

## Si la aplicación no arranca

Comprueba lo siguiente:

- Python 3.12 está instalado
- Los archivos de modelos están colocados en las carpetas correctas
- `setup_env.bat` terminó sin errores
- La ruta de la carpeta no contiene restricciones de permisos inusuales

# PBR Texture Generation and Aging Tool

A local web-based tool for **PBR texture generation and synthetic aging** focused on **stone-like materials** such as **brick, concrete, stone and asphalt**.

This project was developed as an academic prototype and is designed to run **locally on Windows** through **Streamlit**, using **TensorFlow/Keras** pipelines for inference.

---

## Table of Contents

- [English](#english)
  - [1. Project Overview](#1-project-overview)
  - [2. Main Features](#2-main-features)
  - [3. Technology Stack](#3-technology-stack)
  - [4. Project Structure](#4-project-structure)
  - [5. System Requirements](#5-system-requirements)
  - [6. Model Files](#6-model-files)
  - [7. Installation](#7-installation)
  - [8. How to Run the Application](#8-how-to-run-the-application)
  - [9. Recommended Usage Workflow](#9-recommended-usage-workflow)
  - [10. Current Limitations](#10-current-limitations)
  - [11. Repository Scope](#11-repository-scope)
  - [12. Future Work](#12-future-work)
- [Español](#español)
  - [1. Descripción general del proyecto](#1-descripción-general-del-proyecto)
  - [2. Funcionalidades principales](#2-funcionalidades-principales)
  - [3. Stack tecnológico](#3-stack-tecnológico)
  - [4. Estructura del proyecto](#4-estructura-del-proyecto)
  - [5. Requisitos del sistema](#5-requisitos-del-sistema)
  - [6. Archivos de modelos](#6-archivos-de-modelos)
  - [7. Instalación](#7-instalación)
  - [8. Cómo ejecutar la aplicación](#8-cómo-ejecutar-la-aplicación)
  - [9. Flujo de uso recomendado](#9-flujo-de-uso-recomendado)
  - [10. Limitaciones actuales](#10-limitaciones-actuales)
  - [11. Alcance del repositorio](#11-alcance-del-repositorio)
  - [12. Trabajo futuro](#12-trabajo-futuro)

---

# English

## 1. Project Overview

**PBR Texture Generation and Aging Tool** is a local inference application that receives a photograph of a stone-like material and generates physically based rendering (PBR) maps.

The current pipeline is aimed at **restricted-domain materials**, specifically:

- Brick
- Concrete
- Stone
- Asphalt
- Similar mineral or masonry surfaces

The system is intentionally **not general-purpose**. It was designed around a constrained material domain in order to obtain more coherent responses from the neural networks.

### Current pipeline

1. The user uploads a material image.
2. The application validates and previews the image.
3. A **DeepPBR** inference pipeline generates:
   - **Normal map**
   - **Roughness map**
4. Optionally, a **CycleGAN** inference pipeline applies synthetic aging and produces:
   - **Aged albedo**
   - **Aged normal**
   - **Aged roughness**
5. The application displays the generated results and allows downloading them individually or as a ZIP package.

This project is currently a **functional MVP (Minimum Viable Product)** for local academic demonstration and technical validation.

---

## 2. Main Features

The current version includes:

- Local web interface built with **Streamlit**
- Input image upload and preview
- 2x2 tileable preview of the original texture
- Automatic image validation
- Estimated processing time before execution
- **DeepPBR** real patch-based inference with overlap and blending
- Optional **CycleGAN** real-scale patch-based aging pipeline
- Side-by-side visualization of generated outputs
- Interactive before/after comparison slider
- Individual download buttons for generated maps
- ZIP export with all generated results
- Model caching using `@st.cache_resource`

---

## 3. Technology Stack

### Core software

- **Python 3.12.10**
- **TensorFlow 2.19.0**
- **Keras 3.10.0**
- **Streamlit**
- **OpenCV**
- **NumPy**
- **Pillow**
- **Matplotlib**

### Execution environment used during development

- **Windows 11**
- Local execution
- CPU inference validated

---

## 4. Project Structure

Recommended repository structure:

```text
pbr-texture-tool/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── cyclegan_infer.py
│   ├── deeppbr_infer.py
│   ├── geometry.py
│   ├── io_utils.py
│   ├── pipeline_runtime.py
│   ├── preview.py
│   ├── tiling.py
│   └── zip_utils.py
├── models/
│   ├── deeppbr/
│   │   └── (place DeepPBR checkpoint files here)
│   └── cyclegan/
│       └── (place CycleGAN weight files here)
├── outputs/
├── sample_inputs/
├── scripts/
├── app.py
├── requirements.txt
├── setup_env.bat
├── run_app.bat
├── quick_start.md
├── .gitignore
└── README.md
```

### Important note about this repository

This repository is intended to store:

- application code,
- configuration,
- documentation,
- helper scripts,
- and lightweight example assets.

It **was not recommended** to store the large trained models directly in the GitHub repository, especially the DeepPBR checkpoint, because one of its files is approximately **1.6 GB**.

---

## 5. System Requirements

Minimum practical requirements for local execution:

- **Windows 10 or Windows 11**
- **Python 3.12.x** installed and available
- At least **16 GB RAM** recommended
- Enough free disk space for:
  - the project folder,
  - the virtual environment,
  - the model files,
  - generated outputs

---

## 6. Model Files

This repository expects the model files to be placed manually in the following locations.

### DeepPBR

Expected folder:

```text
models/deeppbr/
```

Expected checkpoint prefix in code:

```text
models/deeppbr/ckpt-42
```

That means the DeepPBR files should include at least:

```text
models/deeppbr/checkpoint
models/deeppbr/ckpt-42.index
models/deeppbr/ckpt-42.data-00000-of-00001
```

### CycleGAN

Expected folder:

```text
models/cyclegan/
```

Expected main weights file in code:

```text
models/cyclegan/gen_AB_epoca80_FINAL.weights.h5
```

Additional CycleGAN files may also be kept in the same folder if needed.

---

## 7. Installation

### Option A

Use the provided batch file:

```bat
setup_env.bat
```

This script:

1. checks that Python is installed,
2. creates a local virtual environment named `.venv`,
3. upgrades `pip`,
4. installs dependencies from `requirements.txt`.

### Option B - manual installation

Open PowerShell inside the project folder and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 8. How to Run the Application

### Recommended

Double-click:

```bat
run_app.bat
```

This launches:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Manual execution

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If everything is correctly installed, Streamlit will open a local address in the browser.

---

## 9. Recommended Usage Workflow

1. Place the model files in the expected folders.
2. Run `setup_env.bat` once.
3. Run `run_app.bat`.
4. Upload a front-facing image of a stone-like material.
5. Review the tileable preview and validation warnings.
6. Generate the base maps.
7. Optionally enable synthetic aging.
8. Review the generated outputs.
9. Download the individual maps or the ZIP package.

### Practical advice for better results

- Use images with **frontal perspective** whenever possible.
- Avoid very strong shadows.
- Avoid blurry photographs.
- Avoid materials outside the intended domain.
- Avoid extremely high resolutions on modest hardware.

---

## 10. Current Limitations

This repository represents a **functional academic prototype**, not a finished production product.

Known limitations include:

- Restricted domain: stone-like materials only
- Input quality strongly affects the outputs
- Very large images may increase RAM and CPU usage significantly
- Perspective correction is not yet integrated in the final UI workflow
- The PBR 3D viewer is not yet integrated
- CycleGAN aging quality may vary depending on the input material
- Some residual artifacts may still appear in specific aged maps

---

## 11. Future Work

Possible next steps include:

- interactive perspective correction inside Streamlit
- embedded Three.js PBR viewer
- repository and packaging refinements
- more robust UX feedback
- histogram visualization of generated maps
- better deployment packaging for external users

---

# Español

## 1. Descripción general del proyecto

**PBR Texture Generation and Aging Tool** es una aplicación local con interfaz web que recibe una fotografía de un material pétreo y genera mapas PBR.

El pipeline actual está orientado a un **dominio restringido de materiales**, en concreto:

- Ladrillo
- Hormigón
- Piedra
- Asfalto
- Superficies similares de naturaleza mineral o de fábrica

El sistema **no es generalista**. Se diseñó deliberadamente sobre un dominio acotado para obtener respuestas más coherentes por parte de las redes neuronales.

### Pipeline actual

1. El usuario sube una imagen de material.
2. La aplicación valida y previsualiza la imagen.
3. Un pipeline de inferencia **DeepPBR** genera:
   - **Normal map**
   - **Roughness map**
4. Opcionalmente, un pipeline de inferencia **CycleGAN** aplica envejecimiento sintético y produce:
   - **Albedo aged**
   - **Normal aged**
   - **Roughness aged**
5. La aplicación muestra los resultados generados y permite descargarlos individualmente o en un ZIP.

Actualmente, el proyecto debe entenderse como un **MVP funcional** para demostración académica y validación técnica en local.

---

## 2. Funcionalidades principales

La versión actual incluye:

- Interfaz web local construida con **Streamlit**
- Subida y previsualización de imagen
- Vista previa tileable 2x2 de la textura original
- Validación automática de imagen
- Estimación de tiempo de procesamiento antes de ejecutar
- Inferencia real de **DeepPBR** por parches, con solape y blending
- Pipeline opcional de envejecimiento **CycleGAN** a escala real por parches
- Visualización lado a lado de los resultados generados
- Comparador interactivo before/after con slider
- Botones de descarga individual para los mapas generados
- Exportación ZIP con todos los resultados
- Cacheado de modelos mediante `@st.cache_resource`

---

## 3. Stack tecnológico

### Software principal

- **Python 3.12.10**
- **TensorFlow 2.19.0**
- **Keras 3.10.0**
- **Streamlit**
- **OpenCV**
- **NumPy**
- **Pillow**
- **Matplotlib**

### Entorno de ejecución usado durante el desarrollo

- **Windows 11**
- Ejecución local
- Inferencia en CPU validada

---

## 4. Estructura del proyecto

Estructura recomendada del repositorio:

```text
pbr-texture-tool/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── cyclegan_infer.py
│   ├── deeppbr_infer.py
│   ├── geometry.py
│   ├── io_utils.py
│   ├── pipeline_runtime.py
│   ├── preview.py
│   ├── tiling.py
│   └── zip_utils.py
├── models/
│   ├── deeppbr/
│   │   └── (colocar aquí los checkpoints de DeepPBR)
│   └── cyclegan/
│       └── (colocar aquí los pesos de CycleGAN)
├── outputs/
├── sample_inputs/
├── scripts/
├── app.py
├── requirements.txt
├── setup_env.bat
├── run_app.bat
├── quick_start.md
├── .gitignore
└── README.md
```

### Nota importante sobre este repositorio

Este repositorio está pensado para almacenar:

- código de la aplicación,
- configuración,
- documentación,
- scripts auxiliares,
- y ejemplos ligeros.

**No era recomendable** almacenar directamente en GitHub los modelos entrenados más pesados, especialmente el checkpoint de DeepPBR, porque uno de sus archivos ocupa aproximadamente **1,6 GB**.

---

## 5. Requisitos del sistema

Requisitos prácticos mínimos para ejecutar el proyecto en local:

- **Windows 10 o Windows 11**
- **Python 3.12.x** instalado y disponible
- Se recomiendan al menos **16 GB de RAM**
- Espacio libre suficiente en disco para:
  - la carpeta del proyecto,
  - el entorno virtual,
  - los archivos de modelos,
  - los resultados generados

---

## 6. Archivos de modelos

Este repositorio espera que los archivos de modelos se coloquen manualmente en las siguientes rutas.

### DeepPBR

Carpeta esperada:

```text
models/deeppbr/
```

Prefijo de checkpoint esperado por el código:

```text
models/deeppbr/ckpt-42
```

Eso significa que DeepPBR debería incluir al menos:

```text
models/deeppbr/checkpoint
models/deeppbr/ckpt-42.index
models/deeppbr/ckpt-42.data-00000-of-00001
```

### CycleGAN

Carpeta esperada:

```text
models/cyclegan/
```

Archivo principal de pesos esperado por el código:

```text
models/cyclegan/gen_AB_epoca80_FINAL.weights.h5
```

Los demás archivos de CycleGAN pueden mantenerse en esa misma carpeta si se desea.

---

## 7. Instalación

### Opción A

Usar el archivo `.bat` incluido:

```bat
setup_env.bat
```

Este script:

1. comprueba que Python está instalado,
2. crea un entorno virtual local llamado `.venv`,
3. actualiza `pip`,
4. instala las dependencias desde `requirements.txt`.

### Opción B - instalación manual

Abrir PowerShell dentro de la carpeta del proyecto y ejecutar:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 8. Cómo ejecutar la aplicación

### Recomendado

Hacer doble clic en:

```bat
run_app.bat
```

Este script lanza:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Ejecución manual

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Si todo está correctamente instalado, Streamlit abrirá una dirección local en el navegador.

---

## 9. Flujo de uso recomendado

1. Colocar los archivos de modelos en las carpetas esperadas.
2. Ejecutar `setup_env.bat` una vez.
3. Ejecutar `run_app.bat`.
4. Subir una imagen frontal de un material pétreo.
5. Revisar la preview tileable y los avisos de validación.
6. Generar los mapas base.
7. Activar opcionalmente el envejecimiento sintético.
8. Revisar los resultados generados.
9. Descargar los mapas individuales o el ZIP.

### Consejos prácticos para obtener mejores resultados

- Utilizar imágenes lo más frontales posible.
- Evitar sombras muy duras.
- Evitar fotografías borrosas.
- Evitar materiales fuera del dominio previsto.
- Evitar resoluciones extremadamente altas en equipos modestos.

---

## 10. Limitaciones actuales

Este repositorio representa un **prototipo académico funcional**, no un producto final de producción.

Limitaciones conocidas:

- Dominio restringido: solo materiales pétreos
- La calidad de entrada afecta mucho a la salida
- Las imágenes muy grandes pueden aumentar bastante el uso de RAM y CPU
- La corrección de perspectiva aún no está integrada en el flujo final de la UI
- El visor 3D PBR aún no está integrado
- La calidad del envejecimiento con CycleGAN puede variar según el material de entrada
- Todavía pueden aparecer artefactos residuales en algunos mapas aged

---

## 11. Trabajo futuro

Posibles pasos siguientes:

- corrección de perspectiva interactiva dentro de Streamlit
- visor PBR integrado con Three.js
- mejoras de empaquetado y estructura del repositorio
- feedback UX más robusto
- histogramas de los mapas generados
- mejor empaquetado para usuarios externos


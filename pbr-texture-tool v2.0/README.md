# PBR Texture Tool

A local web-based tool for **automatic PBR map generation and synthetic aging** from photographs of stone-like materials such as brick, concrete, stone and asphalt.

Developed as an academic prototype, the tool runs entirely on a local machine through a **Streamlit** interface, using **TensorFlow/Keras** deep learning pipelines for inference.

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
  - [8. How to Run](#8-how-to-run)
  - [9. Recommended Workflow](#9-recommended-workflow)
  - [10. Known Limitations](#10-known-limitations)
- [Español](#español)
  - [1. Descripción general](#1-descripción-general)
  - [2. Funcionalidades principales](#2-funcionalidades-principales)
  - [3. Stack tecnológico](#3-stack-tecnológico)
  - [4. Estructura del proyecto](#4-estructura-del-proyecto)
  - [5. Requisitos del sistema](#5-requisitos-del-sistema)
  - [6. Archivos de modelos](#6-archivos-de-modelos)
  - [7. Instalación](#7-instalación)
  - [8. Cómo ejecutar](#8-cómo-ejecutar)
  - [9. Flujo de uso recomendado](#9-flujo-de-uso-recomendado)
  - [10. Limitaciones conocidas](#10-limitaciones-conocidas)

---

# English

## 1. Project Overview

**PBR Texture Tool** takes a photograph of a stone-like surface and automatically generates the three maps required for physically based rendering: **Normal map**, **Roughness map** and, optionally, a synthetically aged version of the complete PBR stack.

The pipeline is built around two custom-trained deep learning models:

- **DeepPBR-Net** — a dual-head generator with a ResNet50 encoder and independent CBAM-attention decoders, trained on ~1,500 purified 1K textures from MatSynth. Produces Normal and Roughness maps via overlapping patch inference with Hann-window blending.
- **CycleGAN (7-channel)** — an unpaired image-to-image translation network trained on AmbientCG data, operating on the full PBR stack (RGB + Normal + Roughness). Transforms clean materials into synthetically weathered versions. Includes AdaIN-based intensity control and post-process normal map correction.

The generated maps are ready for direct import into Unreal Engine 5, Unity or any PBR-compatible renderer.

The tool is intentionally **domain-restricted** to stone-family materials. It is not designed as a general-purpose texture generator.

---

## 2. Main Features

**Input and preprocessing**
- Image upload with automatic validation (resolution, aspect ratio)
- Interactive perspective correction with a 4-point draggable canvas — frame any flat surface from an oblique photograph before inference
- Tileable 2×2 preview with seamlessness guidance

**PBR generation**
- DeepPBR-Net inference: Normal map + Roughness map
- Full-resolution patch-based processing with configurable overlap and Hann-window blending
- Zoom factor control to adjust the effective field of view per tile, compensating for very high-resolution inputs

**Synthetic aging (optional)**
- CycleGAN 7-channel aging: aged Albedo, Normal and Roughness
- Adjustable intensity slider (0.1–1.0) via post-process blending between original and aged output
- Post-process histogram matching to correct normal map colour space after aging

**Visualisation and inspection**
- Interactive before/after comparison slider
- Embedded 3D PBR viewer (Three.js) with orbital light control and zoom — supports both base and aged map sets
- In-viewer sliders to adjust normal strength and roughness multiplier in real time
- Optional technical inspection panel with per-channel histograms for Normal and Roughness maps

**Export**
- Individual download buttons organised by map type (Albedo / Normal / Roughness)
- Full ZIP export with all generated maps
- Model caching via `@st.cache_resource` for fast repeated runs

---

## 3. Technology Stack

| Layer | Technology |
|---|---|
| Interface | Streamlit |
| Deep learning | TensorFlow 2.19.0 · Keras 3.10.0 |
| Image processing | OpenCV · Pillow · NumPy |
| 3D viewer | Three.js r128 (CDN) |
| Visualisation | Matplotlib |
| Platform | Windows 10/11 · Python 3.12.x |

---

## 4. Project Structure

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
│   ├── deeppbr/          ← place DeepPBR checkpoint files here
│   └── cyclegan/         ← place CycleGAN weight file here
├── sample_inputs/        ← example textures for testing
├── scripts/
│   ├── check_environment.py
│   ├── validate_cyclegan.py
│   └── validate_deeppbr.py
├── app.py
├── requirements.txt
├── setup_env.bat
├── run_app.bat
├── QUICK_START.md
└── README.md
```

Model files are not stored in this repository due to their size (~1.6 GB for the DeepPBR checkpoint). The complete package including model weights is available in the [Releases](../../releases) section.

---

## 5. System Requirements

- Windows 10 or Windows 11
- Python 3.12.x
- At least 16 GB RAM recommended
- Sufficient disk space for the virtual environment, model files and generated outputs
- GPU optional but recommended for faster inference (CPU execution is fully supported)

---

## 6. Model Files

### DeepPBR

Expected path prefix: `models/deeppbr/ckpt-42`

Required files:
```
models/deeppbr/checkpoint
models/deeppbr/ckpt-42.index
models/deeppbr/ckpt-42.data-00000-of-00001
```

### CycleGAN

Expected file: `models/cyclegan/gen_AB_epoca80_FINAL.weights.h5`

Required files:
```
models/deeppbr/gen_AB_epoca80_FINAL
models/deeppbr/gen_AB_epoca80_FINAL.weights
models/deeppbr/gen_AB_latest.weights
```

---

## 7. Installation

**Option A — automatic**

Double-click `setup_env.bat`. This creates a `.venv` virtual environment and installs all dependencies from `requirements.txt`.

**Option B — manual**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

To verify the environment after installation:
```powershell
.\.venv\Scripts\python.exe scripts/check_environment.py
```

---

## 8. How to Run

Double-click `run_app.bat`, or from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will open the application in the default browser at `http://localhost:8501`.

---

## 9. Recommended Workflow

1. Place model files in `models/deeppbr/` and `models/cyclegan/`.
2. Run `setup_env.bat` once to set up the environment.
3. Launch the app with `run_app.bat`.
4. Upload a front-facing photograph of a stone-like material.
5. *(Optional)* Use the perspective correction tool to frame the flat surface before inference.
6. Review the tileable preview to assess seamlessness.
7. Adjust the zoom factor if needed (recommended for images above 1K resolution).
8. Click **Generate PBR maps**.
9. *(Optional)* Enable synthetic aging and adjust intensity.
10. Inspect results with the before/after slider and the 3D PBR viewer.
11. Download individual maps or the full ZIP.

**Tips for best results**
- Use photographs taken straight-on with even, diffuse lighting.
- Avoid strong shadows, reflections, blur or heavy noise.
- Keep inputs within the intended domain: brick, concrete, stone, asphalt.
- For high-resolution inputs (2K+), reduce the zoom factor so each tile covers a complete, recognisable region of the material.

---

## 10. Known Limitations

- Domain restricted to stone-like materials — results on wood, metal, fabric or organic surfaces are not reliable.
- Input quality directly affects output quality.
- CycleGAN aging produces a characteristic cool colour shift on the albedo; this is a model-level behaviour inherent to the training domain.
- Residual artifacts may appear in aged maps under certain inputs.
- Batch processing is not supported.
- Very high resolutions significantly increase processing time and RAM usage.

---

# Español

## 1. Descripción general

**PBR Texture Tool** toma una fotografía de una superficie pétrea y genera automáticamente los tres mapas necesarios para el renderizado físicamente correcto: **Normal map**, **Roughness map** y, de forma opcional, una versión envejecida sintéticamente del stack PBR completo.

El pipeline se basa en dos modelos de deep learning entrenados a medida:

- **DeepPBR-Net** — generador de doble cabeza con encoder ResNet50 y decoders independientes con atención CBAM, entrenado con ~1.500 texturas 1K purificadas de MatSynth. Genera Normal y Roughness mediante inferencia por parches solapados con blending por ventana de Hann.
- **CycleGAN (7 canales)** — red de traducción imagen-a-imagen no supervisada entrenada con datos de AmbientCG, que opera sobre el stack PBR completo (RGB + Normal + Roughness). Transforma materiales limpios en versiones desgastadas sintéticamente. Incluye control de intensidad mediante AdaIN y corrección post-proceso del espacio de color del normal map.

Los mapas generados están listos para importarse directamente en Unreal Engine 5, Unity o cualquier motor compatible con PBR.

La herramienta está deliberadamente **restringida al dominio pétreo** y no está diseñada como generador de texturas generalista.

---

## 2. Funcionalidades principales

**Entrada y preprocesado**
- Subida de imagen con validación automática (resolución, proporciones)
- Corrección de perspectiva interactiva con canvas de 4 puntos arrastrables — encuadra cualquier superficie plana desde una fotografía oblicua antes de la inferencia
- Preview tileable 2×2 con indicaciones sobre la costura

**Generación PBR**
- Inferencia DeepPBR-Net: Normal map + Roughness map
- Procesado a resolución completa por parches con solape configurable y blending por ventana de Hann
- Control de factor de zoom para ajustar el campo visual efectivo por parche, compensando entradas de muy alta resolución

**Envejecimiento sintético (opcional)**
- Envejecimiento CycleGAN 7 canales: Albedo, Normal y Roughness envejecidos
- Slider de intensidad ajustable (0.1–1.0) mediante mezcla post-proceso entre la salida original y la envejecida
- Corrección de espacio de color del normal map aged mediante histogram matching estadístico

**Visualización e inspección**
- Comparador interactivo before/after con slider
- Visor 3D PBR embebido (Three.js) con control orbital de la luz y zoom — compatible con mapas base y envejecidos
- Sliders en el visor para ajustar la fuerza del normal map y el multiplicador de roughness en tiempo real
- Panel opcional de inspección técnica con histogramas por canal para los mapas Normal y Roughness

**Exportación**
- Botones de descarga individual organizados por tipo de mapa (Albedo / Normal / Roughness)
- Exportación ZIP completa con todos los mapas generados
- Cacheado de modelos con `@st.cache_resource` para ejecuciones repetidas sin recarga

---

## 3. Stack tecnológico

| Capa | Tecnología |
|---|---|
| Interfaz | Streamlit |
| Deep learning | TensorFlow 2.19.0 · Keras 3.10.0 |
| Procesado de imagen | OpenCV · Pillow · NumPy |
| Visor 3D | Three.js r128 (CDN) |
| Visualización | Matplotlib |
| Plataforma | Windows 10/11 · Python 3.12.x |

---

## 4. Estructura del proyecto

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
│   ├── deeppbr/          ← colocar aquí los checkpoints de DeepPBR
│   └── cyclegan/         ← colocar aquí el archivo de pesos de CycleGAN
├── sample_inputs/        ← texturas de ejemplo para pruebas
├── scripts/
│   ├── check_environment.py
│   ├── validate_cyclegan.py
│   └── validate_deeppbr.py
├── app.py
├── requirements.txt
├── setup_env.bat
├── run_app.bat
├── QUICK_START.md
└── README.md
```

Los archivos de modelos no se almacenan en este repositorio por su tamaño (~1,6 GB el checkpoint de DeepPBR). El paquete completo con los pesos está disponible en la sección [Releases](../../releases).

---

## 5. Requisitos del sistema

- Windows 10 o Windows 11
- Python 3.12.x
- Al menos 16 GB de RAM recomendados
- Espacio libre suficiente para el entorno virtual, los modelos y los resultados generados
- GPU opcional pero recomendada para inferencia más rápida (CPU completamente soportada)

---

## 6. Archivos de modelos

### DeepPBR

Prefijo de checkpoint esperado: `models/deeppbr/ckpt-42`

Archivos necesarios:
```
models/deeppbr/checkpoint
models/deeppbr/ckpt-42.index
models/deeppbr/ckpt-42.data-00000-of-00001
```

### CycleGAN

Archivo esperado: `models/cyclegan/gen_AB_epoca80_FINAL.weights.h5`

Archivos necesarios:
```
models/deeppbr/gen_AB_epoca80_FINAL
models/deeppbr/gen_AB_epoca80_FINAL.weights
models/deeppbr/gen_AB_latest.weights
```

---

## 7. Instalación

**Opción A — automática**

Hacer doble clic en `setup_env.bat`. Crea el entorno virtual `.venv` e instala todas las dependencias desde `requirements.txt`.

**Opción B — manual**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Para verificar el entorno tras la instalación:
```powershell
.\.venv\Scripts\python.exe scripts/check_environment.py
```

---

## 8. Cómo ejecutar

Hacer doble clic en `run_app.bat`, o desde PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit abrirá la aplicación en el navegador en `http://localhost:8501`.

---

## 9. Flujo de uso recomendado

1. Colocar los archivos de modelos en `models/deeppbr/` y `models/cyclegan/`.
2. Ejecutar `setup_env.bat` una vez para preparar el entorno.
3. Lanzar la app con `run_app.bat`.
4. Subir una fotografía frontal de un material pétreo.
5. *(Opcional)* Usar la herramienta de corrección de perspectiva para encuadrar la superficie plana antes de la inferencia.
6. Revisar la preview tileable para evaluar la costura.
7. Ajustar el factor de zoom si es necesario (recomendado para imágenes superiores a 1K).
8. Pulsar **Generate PBR maps**.
9. *(Opcional)* Activar el envejecimiento sintético y ajustar la intensidad.
10. Inspeccionar los resultados con el comparador y el visor 3D PBR.
11. Descargar los mapas individuales o el ZIP completo.

**Consejos para mejores resultados**
- Usar fotografías tomadas frontalmente con iluminación difusa y uniforme.
- Evitar sombras duras, reflejos, desenfoque o ruido excesivo.
- Mantener las entradas dentro del dominio previsto: ladrillo, hormigón, piedra, asfalto.
- Para entradas de alta resolución (2K o más), reducir el factor de zoom para que cada parche cubra una región completa y reconocible del material.

---

## 10. Limitaciones conocidas

- Dominio restringido a materiales pétreos — los resultados sobre madera, metal, tela o superficies orgánicas no son fiables.
- La calidad de la entrada afecta directamente a la calidad de la salida.
- El envejecimiento CycleGAN produce un desplazamiento cromático frío característico en el albedo; es un comportamiento del modelo inherente al dominio de entrenamiento.
- Pueden aparecer artefactos residuales en mapas aged bajo ciertas entradas.
- No se soporta procesado por lotes.
- Las resoluciones muy altas aumentan considerablemente el tiempo de procesado y el uso de RAM.

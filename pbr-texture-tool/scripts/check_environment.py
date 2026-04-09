from pathlib import Path
import platform
import sys

import tensorflow as tf
import keras

# ------------------------------------------------------------------
# Raíz del proyecto.
# Este script vive en:
#   C:\pbr_tool\scripts\check_environment.py
# Por eso parents[1] sube a:
#   C:\pbr_tool
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Rutas importantes del proyecto.
DEEPPBR_DIR = PROJECT_ROOT / "models" / "deeppbr"
CYCLEGAN_DIR = PROJECT_ROOT / "models" / "cyclegan"
SAMPLE_INPUTS_DIR = PROJECT_ROOT / "sample_inputs"


def main() -> None:
    """Ejecuta una comprobación básica del entorno local."""

    print("=" * 70)
    print("COMPROBACIÓN DEL ENTORNO LOCAL")
    print("=" * 70)

    # --------------------------------------------------------------
    # Información general del sistema y del intérprete.
    # --------------------------------------------------------------
    print(f"Sistema operativo: {platform.system()} {platform.release()}")
    print(f"Versión de Python: {sys.version}")
    print(f"Ejecutable de Python en uso: {sys.executable}")

    # --------------------------------------------------------------
    # Información de TensorFlow y Keras.
    # --------------------------------------------------------------
    print(f"TensorFlow: {tf.__version__}")
    print(f"Keras: {keras.__version__}")

    # --------------------------------------------------------------
    # Dispositivos detectados por TensorFlow.
    # En esta fase, que solo aparezca CPU no invalida nada.
    # --------------------------------------------------------------
    cpus = tf.config.list_physical_devices("CPU")
    gpus = tf.config.list_physical_devices("GPU")

    print(f"CPUs detectadas por TensorFlow: {cpus}")
    print(f"GPUs detectadas por TensorFlow: {gpus}")

    # --------------------------------------------------------------
    # Comprobación de carpetas importantes.
    # --------------------------------------------------------------
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"¿Existe models/deeppbr? {DEEPPBR_DIR.exists()}")
    print(f"¿Existe models/cyclegan? {CYCLEGAN_DIR.exists()}")
    print(f"¿Existe sample_inputs? {SAMPLE_INPUTS_DIR.exists()}")

    print("=" * 70)
    print("FIN DE LA COMPROBACIÓN")
    print("=" * 70)


if __name__ == "__main__":
    main()

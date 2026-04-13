from pathlib import Path

# -------------------------------------------------------------
# Raíz del proyecto.
# Este archivo vive en:
#   C:\pbr_tool\app\config.py
# Por eso parents[1] sube a:
#   C:\pbr_tool
# -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -------------------------------------------------------------
# Rutas de modelos.
# -------------------------------------------------------------
DEEPPBR_DIR = PROJECT_ROOT / "models" / "deeppbr"
CYCLEGAN_DIR = PROJECT_ROOT / "models" / "cyclegan"

DEEPPBR_CKPT_PREFIX = DEEPPBR_DIR / "ckpt-42"
CYCLEGAN_WEIGHTS = CYCLEGAN_DIR / "gen_AB_epoca80_FINAL.weights.h5"

# -------------------------------------------------------------
# Rutas de inputs/outputs.
# -------------------------------------------------------------
SAMPLE_INPUTS_DIR = PROJECT_ROOT / "sample_inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# -------------------------------------------------------------
# Constantes de inferencia.
# -------------------------------------------------------------
MODEL_SIZE = 256
CYCLEGAN_DEFAULT_INTENSITY = 0.7
